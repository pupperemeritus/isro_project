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
from model.mlstm_utils import (
    FeedForward,  # Importing mLSTM components
    mLSTMBlock,
    mLSTMLayer,
    mLSTMLayerConfig,
    xLSTMLargeBlockStack,
    xLSTMLargeConfig,
)


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
        # Assuming y_pred and y_true are of shape (batch_size, output_chunk_length, grid_cells, features)
        return torch.mean((y_pred - y_true) ** 2)


class EnhancedLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2,
        variant: Optional[str] = "XLSTM",  # Changed default to XLSTM
    ):
        super().__init__()
        self.variant = variant

        # Increase network capacity for outlier patterns
        self.hidden_multiplier = 2  # Doubled capacity
        hidden_dim = hidden_dim * self.hidden_multiplier

        self.batch_norm_input = BatchNorm1d(input_dim)

        # mLSTM Config - you can customize this further
        mlstm_config = xLSTMLargeConfig(
            embedding_dim=hidden_dim,  # hidden_dim is embedding dim for mLSTM
            num_heads=8,  # Example, adjust as needed
            num_blocks=num_layers,  # Use num_layers as num_blocks for mLSTM stack
            vocab_size=1,  # Vocab size not relevant for time series, set to 1
            return_last_states=False,  # No states returned in training mode
            mode="train",  # Set mode to train
            weight_mode="single",  # Or "fused" - experiment
        )
        self.mlstm_stack = xLSTMLargeBlockStack(mlstm_config)

        self.batch_norm1 = BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.batch_norm2 = BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout / 2.0)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.activation = nn.GELU()
        self.final_activation = nn.ReLU()

        # Remove TLSTM-specific components
        if self.variant == "XLSTM":  # Simplified variant check
            self.residual_fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale input while preserving outliers
        x_shape = x.shape
        x_reshaped = x.view(-1, x_shape[-1])
        # Removed erroneous conversion from numpy
        x_norm = x_reshaped.float().to(x.device)
        x_norm = x_norm.view(x_shape)

        # Apply batch norm to input
        x_norm = x_norm.transpose(1, 2)
        x_norm = self.batch_norm_input(x_norm)
        x_norm = x_norm.transpose(1, 2)

        # Pass through mLSTM stack
        mlstm_out, _ = self.mlstm_stack(
            x_norm, state=None
        )  # State is None for now - state management needs more consideration
        mlstm_out = mlstm_out[:, -1:, :]  # Take only the last time step output

        # Apply batch norm after mLSTM
        mlstm_out_bn = mlstm_out.transpose(1, 2)
        mlstm_out_bn = self.batch_norm1(mlstm_out_bn)
        mlstm_out_bn = mlstm_out_bn.transpose(1, 2)
        out = self.dropout(mlstm_out_bn)

        out = self.fc1(out)
        out = out.transpose(1, 2)
        out = self.batch_norm2(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        if self.variant == "XLSTM":  # Simplified variant check
            residual = self.residual_fc(x_norm)
            out = out + residual

            # Use ReLU instead of previous final activation to remove negatives
            out = self.final_activation(out)

        return out


class SimpleLogCoshLoss(nn.Module):
    def __init__(self, scale: float = 1.0, epsilon: float = 1e-4):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true
        loss = torch.where(
            torch.abs(diff) < self.epsilon,
            0.5 * diff**2,
            torch.log(torch.cosh(self.scale * diff)) / self.scale,
        )
        return torch.mean(loss)


class CustomLSTMModule(PLPastCovariatesModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = GridMSELoss()

    def configure_optimizers(self) -> Dict:
        # Adjust optimizer for better outlier handling
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.001,  # Increased learning rate
            weight_decay=0.0001,  # Reduced weight decay
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        def lr_lambda(epoch):
            warmup_epochs = 5  # Reduced warmup period
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.65 * (  # Slower decay
                1
                + np.cos(
                    np.pi
                    * (epoch - warmup_epochs)
                    / (self.trainer.max_epochs - warmup_epochs)
                )
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
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

    def _create_model(
        self,
        train_sample: Tuple[
            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
        ],
    ) -> nn.Module:

        # Input dimension is always taken from the sample's feature dimension
        input_dim = train_sample[0].shape[2]  # single input tensor for mLSTM
        output_dim = (
            train_sample[1].shape[2] if train_sample[1] is not None else input_dim
        )
        lstm_kwargs = {
            "input_dim": input_dim,  # corrected input_dim for mLSTM
            "hidden_dim": self.hparams.hidden_dim,
            "num_layers": self.hparams.n_rnn_layers,
            "output_dim": output_dim,
            "dropout": self.hparams.dropout,
            "variant": self.model_variant,
        }
        return EnhancedLSTM(**lstm_kwargs)

    def forward(
        self, inputs: torch.Tensor  # Input is a single tensor for mLSTM
    ) -> torch.Tensor:
        """
        Forward pass for mLSTM, expects single tensor input
        """
        return self.model(inputs)


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
        self._pl_module_class = CustomLSTMModule  # updated module reference


# ----- MAIN: Use the old working approach directly -----
if __name__ == "__main__":
    torch.set_autocast_enabled(True)
    torch.set_float32_matmul_precision("medium")
    # Define parameters
    prediction_horizon = 15
    grid_resolution = 2.5
    every = "15m"
    period = "30m"
    offset = "0m"
    model_variant_type = "XLSTM"  # Changed to only use XLSTM variant

    model_params = {
        "model": "LSTM",
        "input_chunk_length": 45,  # Reduced back to original value
        "output_chunk_length": prediction_horizon,
        "n_rnn_layers": 12,  # Reduced layers
        "hidden_dim": 256,  # Reduced complexity
        "output_chunk_shift": 0,
        "model_variant": model_variant_type,  # Pass model variant to CustomBlockRNNModel
    }
    training_params = {
        "batch_size": 32,
        "n_epochs": 500,  # Reduced epochs
        "optimizer_kwargs": {"lr": 0.001},  # Adjusted learning rate
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
            "gradient_clip_val": 0.5,
            "accumulate_grad_batches": 2,
            "min_epochs": 25,
            "max_epochs": 100,
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
