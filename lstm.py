from typing import Dict, Optional, Tuple, Any

import numpy as np
import polars as pl
import torch
from darts import TimeSeries
from darts.metrics import mae, mape, smape
from darts.models import BlockRNNModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule
import torch.nn as nn
from pytorch_lightning.loggers import MLFlowLogger
import logging
import matplotlib.pyplot as plt
from pytorch_lightning.tuner import Tuner
import torch.nn.functional as F

from model.data_timeseries import preprocess_data, split_data
from model.data_loader import load_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading data...")
data = load_data("/home/pupperemeritus/isro_project/data/October2023.parquet")

# Clear CUDA cache
torch.clear_autocast_cache()
torch.cuda.memory.empty_cache()
torch.cuda.empty_cache()

# Define parameters
prediction_horizon = 15
grid_resolution = 2.5
every = "15m"
period = "30m"
offset = "0m"

# Preprocess data
logger.info("Preprocessing data...")
s4_series, phase_series = preprocess_data(
    data,
    period=period,
    every=every,
    offset=offset,
    lat_range=(-5, 30),
    lon_range=(55, 105),
    grid_resolution=grid_resolution,
)

# Split the data
logger.info("Splitting data...")
s4_train, s4_val, s4_test = split_data(s4_series)
phase_train, phase_val, phase_test = split_data(phase_series)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate the Symmetric Mean Absolute Percentage Error (SMAPE).

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :return: SMAPE value
    """
    # Remove any pairs where both actual and predicted are zero
    mask = ~((actual == 0) & (predicted == 0))
    actual = actual[mask]
    predicted = predicted[mask]

    # Avoid division by zero
    denominator = np.abs(actual) + np.abs(predicted)

    # If denominator is zero, replace with a small number
    denominator = np.where(denominator == 0, 1e-8, denominator)

    smape_val = 2.0 * np.mean(np.abs(predicted - actual) / denominator)
    return smape_val * 100  # Convert to percentage


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :return: MAPE value
    """
    # Remove any pairs where actual is zero
    mask = actual != 0
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        return np.nan

    mape_val = np.mean(np.abs((actual - predicted) / actual))
    return mape_val * 100  # Convert to percentage


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE).

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :return: RMSE value
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate multiple metrics and handle potential issues.

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :return: Dictionary of metric values
    """
    # Ensure inputs are numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        print("WARNING: No valid data points for metric calculation")
        return {"SMAPE": np.nan, "MAPE": np.nan, "RMSE": np.nan}

    metrics = {
        "SMAPE": smape(actual, predicted),
        "MAPE": mape(actual, predicted),
        "RMSE": rmse(actual, predicted),
    }

    # Print diagnostic information
    print(f"Number of valid data points: {len(actual)}")
    print(f"Actual range: {actual.min()} to {actual.max()}")
    print(f"Predicted range: {predicted.min()} to {predicted.max()}")
    print(f"Metrics: {metrics}")

    return metrics


logger.info(
    f"s4_train.all_values(copy=False).shape = {s4_train.all_values(copy=False).shape}"
)
logger.info(
    f"s4_val.all_values(copy=False).shape = {s4_val.all_values(copy=False).shape}"
)
logger.info(
    f"s4_test.all_values(copy=False).shape = {s4_test.all_values(copy=False).shape}"
)
logger.info("Training data shape: %s", s4_train.all_values().shape)
logger.info("Validation data shape: %s", s4_val.all_values().shape)
logger.info("Test data shape: %s", s4_test.all_values().shape)

logger.info(
    "NaN or inf in training data: %s",
    np.any(np.isnan(s4_train.all_values())) or np.any(np.isinf(s4_train.all_values())),
)
logger.info(
    "NaN or inf in validation data: %s",
    np.any(np.isnan(s4_val.all_values())) or np.any(np.isinf(s4_val.all_values())),
)
logger.info(
    "NaN or inf in test data: %s",
    np.any(np.isnan(s4_test.all_values())) or np.any(np.isinf(s4_test.all_values())),
)

logger.info("Training data - first few values:")
logger.info(s4_train.all_values()[1])
logger.info("Training data - statistics:")
logger.info("Min: %s", np.nanmin(s4_train.all_values()))
logger.info("Max: %s", np.nanmax(s4_train.all_values()))
logger.info("Mean: %s", np.nanmean(s4_train.all_values()))
logger.info("Std: %s", np.nanstd(s4_train.all_values()))


class GridMSELoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Assuming y_pred and y_true are of shape (batch_size, output_chunk_length, grid_cells, features)
        return torch.mean((y_pred - y_true) ** 2)


from torch.nn import LayerNorm, Dropout, BatchNorm1d


class EnhancedLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        variant: str = "LSTM",  # "LSTM", "TLSTM", "XLSTM", or "BOTH"
    ):
        super().__init__()
        self.variant = variant
        self.batch_norm_input = BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.batch_norm1 = BatchNorm1d(hidden_dim)
        self.dropout = Dropout(0.1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.batch_norm2 = BatchNorm1d(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        if self.variant in ("TLSTM", "BOTH"):
            self.time_gate = nn.Parameter(torch.tensor(0.5))
        if self.variant in ("XLSTM", "BOTH"):
            self.residual_fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply batch norm to input
        x_norm = x.transpose(1, 2)
        x_norm = self.batch_norm_input(x_norm)
        x_norm = x_norm.transpose(1, 2)

        lstm_out, _ = self.lstm(x_norm)

        # Apply batch norm after LSTM
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = self.batch_norm1(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)

        out = self.dropout(lstm_out)
        out = self.fc1(out)

        # Apply batch norm before final layer
        out = out.transpose(1, 2)
        out = self.batch_norm2(out)
        out = out.transpose(1, 2)
        out = self.fc2(out)

        if self.variant in ("XLSTM", "BOTH"):
            residual = self.residual_fc(x_norm)
            out = out + residual
        if self.variant in ("TLSTM", "BOTH"):
            out = out * torch.sigmoid(self.time_gate)
        return out


class WeightedMSELoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        weights = torch.abs(y_true) / torch.mean(torch.abs(y_true))
        return torch.mean(weights * (y_pred - y_true) ** 2)


class CustomLSTMModule(PLPastCovariatesModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_fn = WeightedMSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.0001,  # Lower initial learning rate
            weight_decay=0.001,  # Reduced weight decay
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Custom learning rate schedule with warmup
        def lr_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.5 * (
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

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

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
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            output_chunk_shift=output_chunk_shift,
            **kwargs,
        )
        self._pl_module_class = CustomLSTMModule


torch.set_float32_matmul_precision("medium")
# Define Darts LSTM model
logger.info("Defining model...")
model = CustomBlockRNNModel(
    model="LSTM",
    input_chunk_length=45,
    output_chunk_length=15,
    n_rnn_layers=24,
    hidden_dim=256,
    output_chunk_shift=0,
    batch_size=32,
    n_epochs=1000,
    optimizer_kwargs={"lr": 0.001, "weight_decay": 0.001},
    model_name="CustomLSTMForecast",
    pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": 1,
        "callbacks": [
            EarlyStopping(
                monitor="val_loss",
                patience=25,
                mode="min",
            ),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="custom-lstm-{epoch:02d}-{val_loss:.5f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        "logger": MLFlowLogger(
            experiment_name="custom_lstm_logs", tracking_uri="mlruns"
        ),
        "precision": "32-true",
        "gradient_clip_val": 0.5,
        "accumulate_grad_batches": 4,
        "min_epochs": 50,
        "max_epochs": 1000,
        "deterministic": True,
        "enable_checkpointing": True,
    },
)

# Train the model
logger.info("Training model...")
model.fit(s4_train, val_series=s4_val)

# Check if the model parameters contain any NaN values
has_nan_params = any(torch.isnan(param).any() for param in model.model.parameters())
logger.info("Model parameters contain NaN: %s", has_nan_params)

# Make predictions
print("Making predictions...")
predictions = model.predict(n=len(s4_test), series=s4_test)
print("Predictions - first few values:")
print(predictions.all_values()[:1])
print("Predictions - statistics:")
print("Min: %s", np.nanmin(predictions.all_values()))
print("Max: %s", np.nanmax(predictions.all_values()))
print("Mean: %s", np.nanmean(predictions.all_values()))
print("Std: %s", np.nanstd(predictions.all_values()))

# Evaluate the model
metrics = calculate_metrics(s4_test.all_values(), predictions.all_values())
