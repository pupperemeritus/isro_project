from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
import torch
from darts import TimeSeries
from darts.metrics import mae, mape, smape
from darts.models import Prophet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
import torch.nn.functional as F

from model.data_timeseries import preprocess_data, split_data
from model.data_loader import IonosphereDataModule, load_data

data = load_data("/home/pupperemeritus/isro_project/data/October2023.parquet")

# Clear CUDA cache
torch.clear_autocast_cache()
torch.cuda.memory.empty_cache()
torch.cuda.empty_cache()

# Define parameters
prediction_horizon = 30
grid_resolution = 5
every = "15m"
period = "30m"
offset = "0m"
# Create data module

s4_series, phase_series = preprocess_data(
    data,
    period=period,
    every=every,
    offset=offset,
    lat_range=(0, 40),
    lon_range=(65, 100),
    grid_resolution=grid_resolution,
)

# Split the data
s4_train, s4_val, s4_test = split_data(s4_series)
phase_train, phase_val, phase_test = split_data(phase_series)

print(
    f"s4_train.all_values(copy=False).shape = {s4_train.all_values(copy=False).shape}"
)
print(f"s4_val.all_values(copy=False).shape = {s4_val.all_values(copy=False).shape}")
print(f"s4_test.all_values(copy=False).shape = {s4_test.all_values(copy=False).shape}")

# Create and train the Prophet model for S4 data
model_s4 = Prophet()
model_s4.fit(s4_train)

# Make predictions
predictions_s4 = model_s4.predict(len(s4_test))

# Evaluate the model
from darts.metrics import mape, smape, mae

print(f"MAPE: {mape(s4_test, predictions_s4)}")
print(f"SMAPE: {smape(s4_test, predictions_s4)}")
print(f"MAE: {mae(s4_test, predictions_s4)}")
