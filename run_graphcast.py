# train_graphcast.py
import logging
from typing import Any, Dict, Optional, Tuple
import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import functools
import time
from darts import TimeSeries
from rich import print
from rich.console import Console
import zarr
import pandas as pd
import polars as pl
import mlflow  # Added for MLFlow logging
import pickle  # Added for checkpoint saving

from model.data_loader import load_data
from model.data_timeseries import (
    preprocess_data as original_preprocess_data,
    split_data,
)
from model.logging_conf import get_logger, setup_logging
from model.metrics import calculate_metrics
from model.graphcast_model import (
    CompactGraphCast,
    CompactGraphCastModelConfig,
    CompactGraphCastTaskConfig,
)
from model.graphcast_dataloader import IonosphereGraphCastDataModule

# Set up logging
root_logger, collector_handler = setup_logging(log_level=logging.INFO)
logger = get_logger(__name__)
console = Console()


# Remove torch-based training code and create a Haiku transform:
def forward_fn(graph_input, model_config, task_config):
    model = CompactGraphCast(
        model_config=CompactGraphCastModelConfig(**model_config),
        task_config=CompactGraphCastTaskConfig(**task_config),
    )
    return model(graph_input)


model_transformed = hk.transform(forward_fn)


# Define loss function (using simple MSE)
def loss_fn(params, rng, batch, model_config, task_config):
    # Assume batch is a tuple (graph_input, target) where features are numpy arrays.
    graph_input, target = batch
    pred = model_transformed.apply(params, rng, graph_input, model_config, task_config)
    loss = jnp.mean((pred - target) ** 2)
    return loss


# Training step with grad computation
@jax.jit
def train_step(params, opt_state, rng, batch, model_config, task_config):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params, rng, batch, model_config, task_config)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# ----- MAIN -----
if __name__ == "__main__":
    # Define parameters
    prediction_horizon = 15
    grid_resolution = 2.5
    every = "15m"
    period = "30m"
    offset = "0m"
    sequence_length = 45

    model_params = {
        "input_chunk_length": sequence_length,
        "output_chunk_length": prediction_horizon,
        "output_chunk_shift": 0,
        "model_config": dataclasses.asdict(
            CompactGraphCastModelConfig(
                resolution=grid_resolution,
                mesh_size=1,
                latent_size=64,
                gnn_msg_steps=4,
                hidden_layers=1,
                radius_query_fraction_edge_length=0.8,
            )
        ),
        "task_config": dataclasses.asdict(
            CompactGraphCastTaskConfig(
                input_variables=["features_s4", "features_phase"],
                target_variables=["target_s4", "target_phase"],
                pressure_levels=[],
                input_duration=every,
            )
        ),
    }
    training_params = {
        "batch_size": 32,
        "n_epochs": 100,
        "optimizer_kwargs": {"lr": 0.0001, "weight_decay": 0.0001},
        "model_name": "CompactGraphCastForecast",
    }
    data_params = {
        "data_path": "/home/pupperemeritus/isro_project/data/October2023.parquet",
        "period": period,
        "every": every,
        "offset": offset,
        "lat_range": (-5, 30),
        "lon_range": (55, 105),
        "grid_resolution": grid_resolution,
        "sequence_length": sequence_length,
        "prediction_horizon": prediction_horizon,
        "stride": 1,
    }

    logger.info("Loading data...")
    data = load_data(data_params["data_path"])
    if data is None:
        print("Data loading failed, exiting.")
        exit()

    # Create IonosphereGraphCastDataModule
    data_module = IonosphereGraphCastDataModule(
        dataframe=data,
        sequence_length=data_params["sequence_length"],
        prediction_horizon=data_params["prediction_horizon"],
        grid_lat_range=data_params["lat_range"],
        grid_lon_range=data_params["lon_range"],
        grid_resolution=data_params["grid_resolution"],
        time_window=data_params["every"],
        stride=data_params["stride"],
        batch_size=training_params["batch_size"],
        num_workers=1,  # Adjust as needed
    )
    data_module.setup()

    # Define model and training parameters:
    model_config = {
        "resolution": grid_resolution,
        "mesh_size": 1,
        "latent_size": 64,
        "gnn_msg_steps": 4,
        "hidden_layers": 1,
        "radius_query_fraction_edge_length": 0.8,
    }
    task_config = {
        "input_variables": ["features_s4", "features_phase"],
        "target_variables": ["target_s4", "target_phase"],
        "pressure_levels": [],
        "input_duration": every,
    }

    rng = jax.random.PRNGKey(42)
    # Get a sample batch from train_dataloader and convert to numpy
    sample_batch = next(iter(data_module.train_dataloader()))

    # Assume sample_batch[0] is a graph with numpy features; if torch tensor, convert via .numpy()
    def to_numpy(x):
        return x.numpy() if hasattr(x, "numpy") else x

    sample_graph = sample_batch[0]
    sample_target = sample_batch[1]
    # Conversion: customize if needed based on your TypedGraph structure
    sample_graph = jax.tree_map(to_numpy, sample_graph)
    sample_target = jnp.array(to_numpy(sample_target))

    logger.info(f"grid_lat_steps: {data_module.grid_lat_steps}")
    logger.info(f"grid_lon_steps: {data_module.grid_lon_steps}")
    logger.info(
        f"num_grid_nodes (product): {data_module.grid_lat_steps * data_module.grid_lon_steps}"
    )
    params = model_transformed.init(rng, sample_graph, model_config, task_config)

    # Initialize optimizer:
    learning_rate = 0.0001
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)

    # Initialize MLFlow logging and early stopping variables
    mlflow.start_run(experiment_id="compact_graphcast_logs")
    mlflow.log_params(
        {
            "grid_resolution": grid_resolution,
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
            "n_epochs": training_params["n_epochs"],
            "learning_rate": learning_rate,
        }
    )
    best_loss = float("inf")
    patience = 10
    patience_counter = 0
    checkpoint_path = "best_model.pkl"

    # Training loop with early stopping
    n_epochs = training_params["n_epochs"]
    logger.info("Starting training...")
    for epoch in range(n_epochs):
        epoch_loss = []
        start_time = time.time()
        for batch in data_module.train_dataloader():
            graph_in, target = batch
            # Convert batch to numpy arrays:
            graph_in = jax.tree_map(to_numpy, graph_in)
            target = jnp.array(to_numpy(target))
            batch_np = (graph_in, target)
            params, opt_state, loss = train_step(
                params, opt_state, rng, batch_np, model_config, task_config
            )
            epoch_loss.append(loss)
        avg_loss = jnp.mean(jnp.array(epoch_loss))
        logger.info(
            f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.5f} - Time: {time.time()-start_time:.2f}s"
        )
        mlflow.log_metric("train_loss", float(avg_loss), step=epoch)
        # Early stopping and checkpoint saving logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            with open(checkpoint_path, "wb") as f:
                pickle.dump(params, f)
            logger.info(
                f"Checkpoint saved at epoch {epoch+1} with loss {best_loss:.5f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    logger.info("Training completed.")
    mlflow.end_run()

    # Prediction (example):
    predictions = model_transformed.apply(
        params, rng, sample_graph, model_config, task_config
    )
    logger.info("Prediction completed.")
    # Optionally save predictions...
    # ...existing code for logging and final outputs...
    collected_logs = collector_handler.get_collected_logs()
    print("\n--- Collected Warnings and Errors Summary ---")
    if collected_logs:
        for level, logs in collected_logs.items():
            if logs:
                print(f"\n--- {level} Logs ---")
                for log_message in logs:
                    print(log_message)
    else:
        print("No warnings or errors were collected during execution.")
    logger.info(f"--- End of run_graphcast.py execution ---")
