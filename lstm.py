from typing import Dict, Optional, Tuple

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

from model.data_timeseries import preprocess_data, split_data
from model.data_loader import load_data

# Load data
data = load_data("/home/pupperemeritus/isro_project/data/October2023.parquet")

# Clear CUDA cache
torch.clear_autocast_cache()
torch.cuda.memory.empty_cache()
torch.cuda.empty_cache()

# Define parameters
prediction_horizon = 30
grid_resolution = 2.5
every = "15m"
period = "30m"
offset = "0m"

# Preprocess data
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
s4_train, s4_val, s4_test = split_data(s4_series)
phase_train, phase_val, phase_test = split_data(phase_series)


def smape(actual, predicted):
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


def mape(actual, predicted):
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


def rmse(actual, predicted):
    """
    Calculate the Root Mean Square Error (RMSE).

    :param actual: Array of actual values
    :param predicted: Array of predicted values
    :return: RMSE value
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_metrics(actual, predicted):
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


print(
    f"s4_train.all_values(copy=False).shape = {s4_train.all_values(copy=False).shape}"
)
print(f"s4_val.all_values(copy=False).shape = {s4_val.all_values(copy=False).shape}")
print(f"s4_test.all_values(copy=False).shape = {s4_test.all_values(copy=False).shape}")
print("Training data shape:", s4_train.all_values().shape)
print("Validation data shape:", s4_val.all_values().shape)
print("Test data shape:", s4_test.all_values().shape)

print(
    "NaN or inf in training data:",
    np.any(np.isnan(s4_train.all_values())) or np.any(np.isinf(s4_train.all_values())),
)
print(
    "NaN or inf in validation data:",
    np.any(np.isnan(s4_val.all_values())) or np.any(np.isinf(s4_val.all_values())),
)
print(
    "NaN or inf in test data:",
    np.any(np.isnan(s4_test.all_values())) or np.any(np.isinf(s4_test.all_values())),
)

print("Training data - first few values:")
print(s4_train.all_values()[1])
print("Training data - statistics:")
print("Min:", np.nanmin(s4_train.all_values()))
print("Max:", np.nanmax(s4_train.all_values()))
print("Mean:", np.nanmean(s4_train.all_values()))
print("Std:", np.nanstd(s4_train.all_values()))


class GridMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        # Assuming y_pred and y_true are of shape (batch_size, output_chunk_length, grid_cells, features)
        return torch.mean((y_pred - y_true) ** 2)


class CustomLSTMModule(PLPastCovariatesModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = GridMSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss


class CustomBlockRNNModel(BlockRNNModel):
    def __init__(
        self,
        model,
        input_chunk_length,
        output_chunk_length,
        hidden_dim,
        n_rnn_layers,
        output_chunk_shift=0,
        **kwargs,
    ):
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


# Define Darts LSTM model
model = CustomBlockRNNModel(
    model="LSTM",
    input_chunk_length=30,
    output_chunk_length=prediction_horizon,
    n_rnn_layers=2,
    hidden_dim=128,
    output_chunk_shift=0,
    batch_size=32,
    n_epochs=50,
    optimizer_kwargs={"lr": 0.001},
    model_name="CustomLSTMForecast",
    pl_trainer_kwargs={
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
            experiment_name="custom_lstm_logs", tracking_uri="mlruns"
        ),
        "precision": "32-true",
        "enable_progress_bar": True,
        "accumulate_grad_batches": 3,
        "min_epochs": 25,
        "deterministic": True,
        "enable_checkpointing": True,
    },
)

# Train the model
model.fit(s4_train, val_series=s4_val)

# Check if the model parameters contain any NaN values
has_nan_params = any(torch.isnan(param).any() for param in model.model.parameters())
print("Model parameters contain NaN:", has_nan_params)

# Make predictions
predictions = model.predict(n=len(s4_test), series=s4_test)
print("Predictions - first few values:")
print(predictions.all_values()[:1])
print("Predictions - statistics:")
print("Min:", np.nanmin(predictions.all_values()))
print("Max:", np.nanmax(predictions.all_values()))
print("Mean:", np.nanmean(predictions.all_values()))
print("Std:", np.nanstd(predictions.all_values()))
# Evaluate the model
metrics = calculate_metrics(s4_test.all_values(), predictions.all_values())
