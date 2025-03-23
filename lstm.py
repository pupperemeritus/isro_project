import hashlib
import logging
import os
import pickle
from pathlib import Path

# mlstm imports
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import zarr
from darts.models import BlockRNNModel
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from rich import print
from rich.console import Console
from torch.nn import BatchNorm1d

from model.data_loader import load_data
from model.data_timeseries import preprocess_data, split_data
from model.logging_conf import get_logger, setup_logging
from model.metrics import calculate_metrics
from rich.table import Table
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm.blocks.mlstm.block import mLSTMBlockConfig
from xlstm.blocks.slstm.block import sLSTMBlockConfig
from xlstm.components.feedforward import FeedForwardConfig
from xlstm.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig


def cache_data(data, cache_path):
    """Cache preprocessed data to file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


def load_cached_data(cache_path):
    """Load preprocessed data from cache."""
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return None


# Set up logging
root_logger, collector_handler = setup_logging(log_level=logging.INFO)
logger = get_logger(__name__)
console = Console()


class GridMSELoss(nn.Module):  # type: ignore[valid-type, misc]
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure both inputs have 4 dimensions: (batch, seq, grid_cells, features).
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(2)  # add grid_cells=1
            y_true = y_true.unsqueeze(2)
        return torch.mean((y_pred - y_true) ** 2)


class SpatialWeightedMSE(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # Weight for point-wise error
        self.beta = beta  # Weight for spatial gradient error

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure inputs have 4 dimensions: (batch, seq, grid_cells, features)
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(2)
            y_true = y_true.unsqueeze(2)

        # Point-wise error
        mse_error = torch.mean((y_pred - y_true) ** 2)

        # Spatial gradient error (captures spatial relationships)
        # Calculate differences across grid cells when applicable
        if y_pred.size(2) > 1:  # Only if we have multiple grid cells
            y_pred_grad = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
            y_true_grad = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
            grad_error = torch.mean((y_pred_grad - y_true_grad) ** 2)
        else:
            grad_error = 0

        return self.alpha * mse_error + self.beta * grad_error


class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=0.05, high_weight=10.0):
        super().__init__()
        self.threshold = threshold
        self.high_weight = high_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure both inputs have 4 dimensions
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(2)
            y_true = y_true.unsqueeze(2)

        # Create weights: higher values get more weight
        weights = torch.ones_like(y_true)
        weights = torch.where(y_true > self.threshold, self.high_weight, weights)

        # Compute weighted MSE
        squared_errors = (y_pred - y_true) ** 2
        weighted_squared_errors = weights * squared_errors

        return torch.mean(weighted_squared_errors)


class CustomLSTMModule(PLPastCovariatesModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # For x-LSTM, you can still switch loss if needed (default here remains WeightedMSELoss)
        loss_type = getattr(self.hparams, "loss_type", "weighted")
        if loss_type == "grid":
            self.loss_fn = GridMSELoss()
        elif loss_type == "spatial":
            self.loss_fn = SpatialWeightedMSE()
        elif loss_type == "weighted":
            self.loss_fn = WeightedMSELoss(threshold=0.05, high_weight=10.0)
        else:
            self.loss_fn = GridMSELoss()
        self._build_lstm_model(self.train_sample)

    def _build_lstm_model(
        self,
        train_sample: Tuple[
            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
        ],
    ) -> None:
        # Restore original x-LSTM model configuration
        input_dim = train_sample[0].shape[2]
        output_dim = (
            train_sample[1].shape[2] if train_sample[1] is not None else input_dim
        )
        #     embedding_dim=512,
        # num_heads=4,
        # num_blocks=6,
        # vocab_size=2048,
        # return_last_states=True,
        # mode="inference",
        # chunkwise_kernel="chunkwise--triton_xl_chunk", # xl_chunk == TFLA kernels
        # sequence_kernel="native_sequence__triton",
        # step_kernel="triton",
        self.batch_norm_input = BatchNorm1d(input_dim)
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=256,
            num_blocks=7,
            embedding_dim=128,
            slstm_at=[1],
        )
        self.mlstm_stack = xLSTMBlockStack(xlstm_config)
        # self.batch_norm1 = BatchNorm1d(self.hparams.hidden_dim)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.fc = nn.Linear(self.hparams.hidden_dim, output_dim)
        self.residual_fc = nn.Linear(input_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        # Initialize output layer weights (as before)
        nn.init.uniform_(self.fc.weight, 0, 1)
        nn.init.constant_(self.fc.bias, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Debug: check input statistics (uncomment for investigation)
        print("Input stats before norm:", x.min(), x.max(), x.mean())
        b, seq, feat = x.shape
        x = x.reshape(b * seq, feat)
        x = self.batch_norm_input(x)
        x = x.reshape(b, seq, feat)
        # Forward through x-LSTM stack
        mlstm_out, _ = self.mlstm_stack(x)
        # Use only the last timestep
        out = mlstm_out
        # Debug: check mlstm output shape
        print("mlstm output shape:", out.shape)
        out = self.batch_norm1(out)
        out = self.dropout(out)
        out = self.fc(out)
        residual = self.residual_fc(x)
        out = out + residual
        # Apply softplus (ensures positive outputs) but check if residual cancels it out
        out = torch.nn.functional.softplus(out)
        return out

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.0005,
            weight_decay=0.01,
        )

        # Simpler learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _calculate_metrics(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        mse = torch.mean((y_hat - y) ** 2)
        mae = torch.mean(torch.abs(y_hat - y))

        # Avoid division by zero
        epsilon = 1e-8
        mape = torch.mean(torch.abs((y - y_hat) / (y + epsilon))) * 100
        smape = (
            torch.mean(
                2.0 * torch.abs(y_hat - y) / (torch.abs(y_hat) + torch.abs(y) + epsilon)
            )
            * 100
        )

        return {"mse": mse, "mae": mae, "mape": mape, "smape": smape}

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss


class CustomBlockRNNModel(BlockRNNModel):
    def __init__(
        self,
        model: str,
        input_chunk_length: int,
        output_chunk_length: int,
        hidden_dim: int,
        n_rnn_layers: int,
        output_chunk_shift: int = 0,
        model_variant: str = "BOTH",
        **kwargs: Any,
    ) -> None:
        self.model_variant = model_variant
        kwargs.pop("model_variant", None)
        super().__init__(
            model=model,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            output_chunk_shift=output_chunk_shift,
            **kwargs,
        )
        self._pl_module_class = CustomLSTMModule  # use restored x-LSTM module


# ----- MAIN: Use the old working approach directly -----
if __name__ == "__main__":
    torch.set_autocast_enabled(True)
    torch.set_float32_matmul_precision("medium")
    # Define parameters
    prediction_horizon = 16
    grid_resolution = 2.5
    every = "15m"
    period = "30m"
    offset = "0m"
    model_variant_type = "XLSTM"  # Changed to only use XLSTM variant

    model_params = {
        "model": "LSTM",
        "input_chunk_length": 64,  # Reduced back to original value
        "output_chunk_length": prediction_horizon,
        "n_rnn_layers": 6,  # Reduced layers
        "hidden_dim": 512,  # Reduced complexity
        "output_chunk_shift": 0,
        "model_variant": model_variant_type,  # Pass model variant to CustomBlockRNNModel
        "loss_type": "grid",  # Options: "grid", "spatial", or "weighted"
    }
    training_params = {
        "batch_size": 32,
        "n_epochs": 500,  # Reduced epochs
        "optimizer_kwargs": {"lr": 0.0005},  # Adjusted learning rate
        "model_name": "CustomLSTMForecast",
        "pl_trainer_kwargs": {
            "accelerator": "gpu",
            "devices": 1,
            "callbacks": [
                EarlyStopping(monitor="val_loss", patience=15, mode="min"),
                ModelCheckpoint(
                    dirpath="checkpoints",
                    filename="custom-lstm-{epoch:02d}-{val_loss:.5f}",
                    save_top_k=3,
                    monitor="val_loss",
                    mode="min",
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            "logger": MLFlowLogger(
                experiment_name="custom_lstm_logs",
                tracking_uri="mlruns",
            ),
            "precision": "32-true",
            "gradient_clip_val": 1,
            "accumulate_grad_batches": 1,
            "min_epochs": 25,
            "max_epochs": 500,
            "deterministic": True,
            "enable_checkpointing": True,
        },
    }
    data_params = {
        "data_path": "/home/pupperemeritus/isro_project/data/October2023.parquet",
        "period": period,
        "every": every,
        "offset": offset,
        "lat_range": (-5, 30),
        "lon_range": (55, 105),
        "grid_resolution": grid_resolution,
    }

    logger.info("Loading data...")
    data = load_data(data_params["data_path"])
    if data is None:
        print("Data loading failed, exiting.")
        exit()

    # Define cache path based on data parameters
    cache_dir = Path("/home/pupperemeritus/isro_project/cache")

    def short_hash(params: Dict[str, Any]) -> str:
        hash_input = "".join(f"{key}:{value}" for key, value in sorted(params.items()))
        return hashlib.md5(hash_input.encode()).hexdigest()[:10]

    cache_filename = (
        f"preprocessed_data_{os.path.basename(data_params['data_path']).split('.')[0]}_"
        f"{short_hash(data_params)}_"
        f".pkl"
    )
    cache_path = cache_dir / cache_filename

    # Try to load from cache first
    cached_data = load_cached_data(cache_path)
    if cached_data is not None:
        logger.info("Loading data from cache...")
        s4_series, phase_series = cached_data
    else:
        logger.info("Loading data...")
        data = load_data(data_params["data_path"])
        if data is None:
            print("Data loading failed, exiting.")
            exit()

        logger.info("Preprocessing data...")
        s4_series, phase_series = preprocess_data(
            data,
            period=data_params["period"],
            every=data_params["every"],
            offset=data_params["offset"],
            lat_range=data_params["lat_range"],
            lon_range=data_params["lon_range"],
            grid_resolution=data_params["grid_resolution"],
        )

        # Cache the preprocessed data
        logger.info("Caching preprocessed data...")
        cache_data((s4_series, phase_series), cache_path)

    logger.info("Splitting data...")
    s4_train, s4_val, s4_test = split_data(s4_series)

    # Use the old working model: instantiate CustomBlockRNNModel directly.
    logger.info(f"Defining model with variant: {model_variant_type}...")
    model = CustomBlockRNNModel(
        model=model_params["model"],
        input_chunk_length=model_params["input_chunk_length"],
        output_chunk_length=model_params["output_chunk_length"],
        n_rnn_layers=model_params["n_rnn_layers"],
        hidden_dim=model_params["hidden_dim"],
        output_chunk_shift=model_params["output_chunk_shift"],
        batch_size=training_params["batch_size"],
        n_epochs=training_params["n_epochs"],
        optimizer_kwargs=training_params["optimizer_kwargs"],
        model_name=training_params["model_name"],
        pl_trainer_kwargs=training_params["pl_trainer_kwargs"],
        model_variant=model_params[
            "model_variant"
        ],  # Pass model variant to CustomBlockRNNModel
    )
    print(model)
    logger.info("Training model...")
    model.fit(s4_train, val_series=s4_val)

    logger.info("Making predictions...")
    predictions = model.predict(n=len(s4_test), series=s4_test)

    # Replace plain prints with rich console prints
    console.print("[bold green]Predictions - statistics:[/bold green]")
    console.print(f"[cyan]Min:[/cyan] {np.nanmin(predictions.all_values())}")
    console.print(f"[cyan]Max:[/cyan] {np.nanmax(predictions.all_values())}")
    console.print(f"[cyan]Mean:[/cyan] {np.nanmean(predictions.all_values())}")
    console.print(f"[cyan]Std:[/cyan] {np.nanstd(predictions.all_values())}")

    # Save predictions as a zarr file
    zarr_path = f"/home/pupperemeritus/isro_project/predictions_{model_variant_type}.zarr"  # Filename includes variant
    zarr.save(zarr_path, predictions.all_values())
    logger.info(
        f"Saved predictions as zarr file to {zarr_path} with variant: {model_variant_type}"
    )

    logger.info("Evaluating model...")
    metrics = calculate_metrics(s4_test.all_values(), predictions.all_values())

    # Log metrics using rich console as a table

    console.print("\n[bold blue]===== Model Evaluation Results =====[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category", style="dim", width=20)
    table.add_column("Value", justify="right")

    for category, values in metrics.items():
        if isinstance(values, dict):
            for name, value in values.items():
                table.add_row(category, name, f"{value:.4f}")
        else:
            table.add_row(category, "", f"{values:.4f}")

    console.print(table)

    # Save metrics to file
    metrics_log_path = (
        f"/home/pupperemeritus/isro_project/metrics_{model_variant_type}.log"
    )
    with open(metrics_log_path, "w") as f:
        for category, values in metrics.items():
            f.write(f"\n{category}:\n")
            if isinstance(values, dict):
                for name, value in values.items():
                    f.write(f"{name}: {value:.4f}\n")
            else:
                f.write(f"{category}: {values:.4f}\n")

    logger.info(f"Detailed metrics saved to {metrics_log_path}")

    # ...existing code for printing collected logs...
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
    logger.info(f"--- End of lstm.py execution with variant: {model_variant_type} ---")
