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
from pytorch_lightning.loggers import MLFlowLogger
import torch.nn.functional as F

from model.data_loader import IonosphereDataModule, load_data

data = load_data("/home/pupperemeritus/isro_project/data/October2023.parquet")

# Clear CUDA cache
torch.clear_autocast_cache()
torch.cuda.memory.empty_cache()
torch.cuda.empty_cache()

# Define parameters
prediction_horizon = 30
grid_resolution = 5
time_window = "5m"
stride = 15

# Create data module
data_module = IonosphereDataModule(
    dataframe=data,
    sequence_length=30,
    prediction_horizon=prediction_horizon,
    grid_lat_range=(0, 40),
    grid_lon_range=(65, 100),
    grid_resolution=grid_resolution,
    time_window=time_window,
    stride=stride,
    batch_size=32,
    num_workers=0,
)

# Prepare data
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()


def loader_to_timeseries(loader):
    all_data_s4 = []
    all_data_phase = []
    all_timestamps = []

    for batch in loader:
        # Combine features and targets for S4
        s4_data = torch.cat([batch["features_s4"], batch["target_s4"]], dim=0)
        s4_data = s4_data.view(
            s4_data.shape[0], -1
        ).numpy()  # Flatten and convert to numpy
        all_data_s4.append(s4_data)

        # Combine features and targets for phase
        phase_data = torch.cat([batch["features_phase"], batch["target_phase"]], dim=0)
        phase_data = phase_data.view(
            phase_data.shape[0], -1
        ).numpy()  # Flatten and convert to numpy
        all_data_phase.append(phase_data)

        # Assuming batch contains timestamps under "IST_Time"
        timestamps = batch["IST_Time"].numpy()  # Convert to numpy array

        # Check the dtype of timestamps
        print("Original timestamps dtype:", timestamps.dtype)

        # Convert numpy.datetime64 to Unix timestamps if necessary
        if np.issubdtype(timestamps.dtype, np.datetime64):
            timestamps_unix = (
                timestamps - np.datetime64("1970-01-01T00:00:00Z")
            ) / np.timedelta64(1, "s")
        else:
            timestamps_unix = timestamps.astype(
                float
            )  # Handle other potential types if needed

        all_timestamps.append(timestamps_unix)

    # Check that all data has the same length
    assert (
        len(all_data_s4) == len(all_data_phase) == len(all_timestamps)
    ), "Mismatch in data lengths"

    # Pad sequences to the same length
    all_data_s4 = pad_to_max_length(all_data_s4)
    all_data_phase = pad_to_max_length(all_data_phase)
    all_timestamps = pad_to_max_length(all_timestamps)

    # Check that all padded data has the same shape
    assert (
        all_data_s4.shape == all_data_phase.shape == all_timestamps.shape
    ), "Mismatch in padded data shapes"

    # Combine all data into Polars DataFrames
    df_s4 = pl.DataFrame(
        {
            "timestamp": all_timestamps.flatten(),
            "S4": all_data_s4.flatten(),
        }
    )

    df_phase = pl.DataFrame(
        {
            "timestamp": all_timestamps.flatten(),
            "Phase": all_data_phase.flatten(),
        }
    )

    # Combine S4 and Phase data into one DataFrame, aligned by timestamp
    combined_df = df_s4.join(df_phase, on="timestamp")

    # Check that the combined DataFrame has the expected number of rows and columns
    expected_rows = len(all_timestamps.flatten())
    expected_cols = 3  # timestamp, S4, Phase
    assert combined_df.shape == (
        expected_rows,
        expected_cols,
    ), f"Combined DataFrame shape mismatch. Expected {(expected_rows, expected_cols)}, got {combined_df.shape}"

    # Convert the Polars DataFrame to a TimeSeries
    return TimeSeries.from_dataframe(combined_df, time_col="timestamp")


def pad_to_max_length(batch_list):
    """
    Pad the batch to the maximum length in the batch_list.
    Convert the padded data to numpy for consistency in Polars.
    """
    max_length = max([tensor.shape[0] for tensor in batch_list])
    padded_batch_list = [
        F.pad(torch.tensor(tensor), (0, 0, 0, max_length - len(tensor))).numpy()
        for tensor in batch_list
    ]
    return padded_batch_list


train_series = loader_to_timeseries(train_loader)
val_series = loader_to_timeseries(val_loader)
test_series = loader_to_timeseries(test_loader)

# Define Darts LSTM model
model = BlockRNNModel(
    model="LSTM",
    input_chunk_length=30,
    output_chunk_length=prediction_horizon,
    n_rnn_layers=2,
    hidden_dim=128,
    batch_size=32,
    n_epochs=50,
    optimizer_kwargs={"lr": 0.001},
    model_name="DartsLSTMForecast",
    pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": 1,
        "callbacks": [
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="darts-lstm-{epoch:02d}-{val_loss:.5f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        "logger": MLFlowLogger(
            experiment_name="darts_lstm_logs", tracking_uri="mlruns"
        ),
        "precision": "16",
        "enable_progress_bar": True,
        "accumulate_grad_batches": 3,
        "profiler": "simple",
        "min_epochs": 25,
        "deterministic": True,
    },
)

# Train the model
model.fit(train_series, val_series=val_series)

# Make predictions
predictions = model.predict(n=len(test_series), series=test_series)

# Evaluate the model
mape_score = mape(test_series, predictions)
smape_score = smape(test_series, predictions)
mae_score = mae(test_series, predictions)

print(f"MAPE: {mape_score}")
print(f"SMAPE: {smape_score}")
print(f"MAE: {mae_score}")
