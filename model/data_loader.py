import logging
import logging.config
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import streamlit as st
import torch
from pytorch_lightning import LightningDataModule
from scipy.interpolate import griddata, interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


logger = logging.Logger(__name__)


def calculate_single_lat_lon(
    elevation: float, azimuth: float, user_lat: float = 17.39, user_long: float = 78.31
):
    logger.debug(
        f"Calculating lat/lon for: el={elevation}, az={azimuth}, user_lat={user_lat}, user_lon={user_long}"
    )
    e_rad = elevation * (np.pi / 180)
    a_rad = azimuth * (np.pi / 180)
    user_lat_rad = user_lat * (np.pi / 180)
    user_long_rad = user_long * (np.pi / 180)

    earth_center_angle = (90 * (np.pi / 180)) - e_rad - np.arcsin(0.94 * np.cos(e_rad))
    earth_center_angle = max(0, earth_center_angle)

    ipp_lat_d1 = np.arcsin(
        (np.sin(user_lat_rad) * np.cos(earth_center_angle))
        + (np.cos(user_lat_rad) * np.sin(earth_center_angle) * np.cos(a_rad))
    )
    ipp_lat = np.degrees(ipp_lat_d1)

    ipp_long_d3 = user_long_rad + np.arcsin(
        (np.sin(earth_center_angle) * np.sin(a_rad)) / np.cos(ipp_lat_d1)
    )
    ipp_lon = np.degrees(ipp_long_d3)

    return ipp_lat, ipp_lon


def add_lat_lon_to_df(
    df: pl.DataFrame,
    elevation_col: str,
    azimuth_col: str,
    user_lat: float,
    user_lon: float,
):
    logger.info("Adding latitude and longitude to dataframe")
    lat, lon = calculate_lat_lon(
        df[elevation_col].to_numpy(), df[azimuth_col].to_numpy(), user_lat, user_lon
    )
    df = df.with_columns([pl.Series("Latitude", lat), pl.Series("Longitude", lon)])
    logger.debug("Latitude and Longitude columns added")
    return df


def gps_to_ist(gps_week: int, gps_seconds: int):
    logger.debug(f"Converting GPS time to IST: week={gps_week}, seconds={gps_seconds}")
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    gps_time = gps_epoch + timedelta(weeks=int(gps_week), seconds=float(gps_seconds))
    ist_time = gps_time + timedelta(hours=5, minutes=30)
    return ist_time


def cached_gps_to_ist(gps_week: int, gps_seconds: int):
    return gps_to_ist(gps_week, gps_seconds)


def calculate_lat_lon(
    elevation: float, azimuth: float, user_lat: float, user_long: float
):
    logger.debug("Calculating lat/lon for multiple points")
    e_rad = np.radians(elevation)
    a_rad = np.radians(azimuth)
    u1_rad = np.radians(user_lat)
    u2_rad = np.radians(user_long)

    earth_center_angle = np.radians(90) - e_rad - np.arcsin(0.94 * np.cos(e_rad))
    earth_center_angle = np.maximum(0, earth_center_angle)

    d1 = np.arcsin(
        (np.sin(u1_rad) * np.cos(earth_center_angle))
        + (np.cos(u1_rad) * np.sin(earth_center_angle) * np.cos(a_rad))
    )
    lat = np.degrees(d1)

    d3 = u2_rad + np.arcsin((np.sin(earth_center_angle) * np.sin(a_rad)) / np.cos(d1))
    lon = np.degrees(d3)

    return lat, lon


def slant_to_vertical(df: pl.DataFrame):
    e_rad = df["Elevation"].to_numpy()
    radius_earth_km = 6378
    height_ipp_km = 350

    p = (
        df[
            "p on Sig1, spectral slope of detrended phase in the 0.1 to 25Hz range (dimensionless)"
        ]
        .cast(pl.Float64)
        .to_numpy()
    )

    amplitude_scintillation_slant = df["S4"].to_numpy()

    term_1 = radius_earth_km * np.cos(e_rad) / (radius_earth_km + height_ipp_km)
    term_2 = np.sqrt(1 - term_1**2)
    term_3 = (1 / term_2) ** (p + 0.25)
    vertical_scintillation_amplitude = pl.Series(
        "Vertical S4", amplitude_scintillation_slant / term_3
    ).fill_nan(np.nanmean(amplitude_scintillation_slant / term_3))

    phase_scintillation_rad = df[
        "Phi60 on Sig1, 60-second phase sigma (radians)"
    ].to_numpy()

    term_4 = (1 / term_2) ** (0.5)
    vertical_scintillation_phase = pl.Series(
        "Vertical Scintillation Phase", phase_scintillation_rad / term_4
    ).fill_nan(np.nanmean(phase_scintillation_rad))

    df.insert_column(
        -1,
        vertical_scintillation_amplitude,
    )
    df.insert_column(-1, vertical_scintillation_phase)

    return df


def preprocess_dataframe(
    df: pl.DataFrame, user_lat: float, user_lon: float
) -> pl.DataFrame:
    df = slant_to_vertical(df)
    df = add_lat_lon_to_df(df, "Elevation", "Azimuth", user_lat, user_lon)
    df = df.with_columns(
        [
            pl.struct(["GPS_WN", "GPS_TOW"])
            .map_elements(
                lambda x: gps_to_ist(x["GPS_WN"], x["GPS_TOW"]), return_dtype=datetime
            )
            .alias("IST_Time")
        ]
    )
    df = df.drop_nulls(
        subset=[
            "GPS_WN",
            "GPS_TOW",
            "SVID",
            "Azimuth",
            "Elevation",
            "S4",
            "Latitude",
            "Longitude",
        ]
    ).drop(
        [
            "GPS_WN",
            "GPS_TOW",
            "Azimuth",
            "Elevation",
            "SVID",
            "sbf2ismr version number",
            "Value of the RxState field of the ReceiverStatus SBF block",
        ]
    )
    # Calculate the null percentage for each column
    null_percentage = df.select(
        [
            (pl.col(c).is_null().sum() / pl.col(c).count() * 100).alias(c)
            for c in df.columns
        ]
    )

    # Find columns with null percentage <= 40%
    valid_columns = [
        col for col in null_percentage.columns if null_percentage[col][0] <= 40
    ]

    # Select only valid columns
    df = df.select(valid_columns)
    return df


def load_data(file: str) -> Optional[pl.DataFrame]:
    try:
        logger.info(f"Starting to load data from file: {file}")
        if file.endswith(".arrow"):
            df = pl.read_ipc(file)
        elif file.endswith(".csv"):
            df = pl.read_csv(file)
        elif file.endswith(".parquet"):
            df = pl.read_parquet(file)
        else:
            raise ValueError("Unsupported file format")

        if df.is_empty():
            logger.warning("Loaded empty DataFrame")
            return None

        required_columns = [
            "WN, GPS Week Number",
            "TOW, GPS Time of Week (seconds)",
            "SVID",
            "Azimuth (degrees)",
            "Elevation (degrees)",
            "Total S4 on Sig1 (dimensionless)",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None

        df = df.rename(
            {
                "WN, GPS Week Number": "GPS_WN",
                "TOW, GPS Time of Week (seconds)": "GPS_TOW",
                "Azimuth (degrees)": "Azimuth",
                "Elevation (degrees)": "Elevation",
                "Total S4 on Sig1 (dimensionless)": "S4",
            }
        )

        df = preprocess_dataframe(df, user_lat=17.39, user_lon=78.31)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Error loading data: {str(e)}")
        return None


class IonosphereDataset(Dataset):

    def __init__(
        self,
        dataframe,
        sequence_length=60,
        prediction_horizon=6,
        grid_lon_range=(65, 100),
        grid_lat_range=(0, 40),
        grid_resolution=1,
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.grid_lon_range = grid_lon_range
        self.grid_lat_range = grid_lat_range
        self.grid_resolution = grid_resolution

        dataframe = dataframe.sort("IST_Time")
        self.timestamps = dataframe["IST_Time"].to_numpy()
        self.longitudes = dataframe["Longitude"].to_numpy()
        self.latitudes = dataframe["Latitude"].to_numpy()

        self.vertical_s4 = np.float32(dataframe["Vertical S4"].to_numpy())
        self.vertical_phase = np.float32(
            dataframe["Vertical Scintillation Phase"].to_numpy()
        )

        self.grid_lon = np.arange(grid_lon_range[0], grid_lon_range[1], grid_resolution)
        self.grid_lat = np.arange(grid_lat_range[0], grid_lat_range[1], grid_resolution)
        self.grid_points = np.array(
            np.meshgrid(self.grid_lon, self.grid_lat)
        ).T.reshape(-1, 2)

        self.minmaxScaler_s4 = MinMaxScaler(feature_range=(0, 1))
        self.minmaxScaler_phase = MinMaxScaler(feature_range=(0, 1))
        self.vertical_s4 = self.minmaxScaler_s4.fit_transform(
            self.vertical_s4.reshape(-1, 1)
        ).flatten()
        self.vertical_phase = self.minmaxScaler_phase.fit_transform(
            self.vertical_phase.reshape(-1, 1)
        ).flatten()

    def __len__(self):
        return len(self.timestamps) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        sequence_data_s4 = [
            self._calculate_grid(
                self.longitudes[idx + i],
                self.latitudes[idx + i],
                self.vertical_s4[idx + i],
            )
            for i in range(self.sequence_length)
        ]
        sequence_data_phase = [
            self._calculate_grid(
                self.longitudes[idx + i],
                self.latitudes[idx + i],
                self.vertical_phase[idx + i],
            )
            for i in range(self.sequence_length)
        ]

        target_data_s4 = [
            self._calculate_grid(
                self.longitudes[idx + self.sequence_length + i],
                self.latitudes[idx + self.sequence_length + i],
                self.vertical_s4[idx + self.sequence_length + i],
            )
            for i in range(self.prediction_horizon)
        ]
        target_data_phase = [
            self._calculate_grid(
                self.longitudes[idx + self.sequence_length + i],
                self.latitudes[idx + self.sequence_length + i],
                self.vertical_phase[idx + self.sequence_length + i],
            )
            for i in range(self.prediction_horizon)
        ]

        return {
            "features_s4": torch.from_numpy(np.array(sequence_data_s4)),
            "features_phase": torch.from_numpy(np.array(sequence_data_phase)),
            "target_s4": torch.from_numpy(np.array(target_data_s4)),
            "target_phase": torch.from_numpy(np.array(target_data_phase)),
        }

    def __getitem__(self, idx):
        # Get the unique timestamps for the sequence and prediction horizon
        timestamps_sequence = self.timestamps[idx : idx + self.sequence_length]
        timestamps_target = self.timestamps[
            idx
            + self.sequence_length : idx
            + self.sequence_length
            + self.prediction_horizon
        ]

        # Group the data by IST_Time for the sequence
        sequence_data_s4 = [
            self._calculate_grid(
                self.longitudes[self.timestamps == timestamp],
                self.latitudes[self.timestamps == timestamp],
                self.vertical_s4[self.timestamps == timestamp],
            )
            for timestamp in timestamps_sequence
        ]

        sequence_data_phase = [
            self._calculate_grid(
                self.longitudes[self.timestamps == timestamp],
                self.latitudes[self.timestamps == timestamp],
                self.vertical_phase[self.timestamps == timestamp],
            )
            for timestamp in timestamps_sequence
        ]

        # Group the data by IST_Time for the prediction horizon
        target_data_s4 = [
            self._calculate_grid(
                self.longitudes[self.timestamps == timestamp],
                self.latitudes[self.timestamps == timestamp],
                self.vertical_s4[self.timestamps == timestamp],
            )
            for timestamp in timestamps_target
        ]

        target_data_phase = [
            self._calculate_grid(
                self.longitudes[self.timestamps == timestamp],
                self.latitudes[self.timestamps == timestamp],
                self.vertical_phase[self.timestamps == timestamp],
            )
            for timestamp in timestamps_target
        ]

        return {
            "features_s4": torch.from_numpy(np.array(sequence_data_s4)),
            "features_phase": torch.from_numpy(np.array(sequence_data_phase)),
            "target_s4": torch.from_numpy(np.array(target_data_s4)),
            "target_phase": torch.from_numpy(np.array(target_data_phase)),
        }


class IonosphereDataModule(LightningDataModule):
    def __init__(
        self,
        dataframe: pl.DataFrame,
        sequence_length: int,
        prediction_horizon: int,
        grid_lon_range: Tuple[float, float],
        grid_lat_range: Tuple[float, float],
        grid_resolution: float,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.dataframe = dataframe
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.grid_lon_range = grid_lon_range
        self.grid_lat_range = grid_lat_range
        range_func = lambda x: int(abs(x[1] - x[0]))
        self.grid_resolution = grid_resolution
        self.grid_size = (range_func(grid_lat_range), range_func(grid_lon_range))
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Split the data into train, validation, and test sets
        total_samples = len(self.dataframe)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)

        train_df = self.dataframe.slice(0, train_size)
        val_df = self.dataframe.slice(train_size, train_size + val_size)
        test_df = self.dataframe.slice(train_size + val_size, total_samples)

        self.train_dataset = IonosphereDataset(
            train_df,
            self.sequence_length,
            self.prediction_horizon,
            self.grid_size,
        )
        self.val_dataset = IonosphereDataset(
            val_df,
            self.sequence_length,
            self.prediction_horizon,
            self.grid_size,
        )
        self.test_dataset = IonosphereDataset(
            test_df,
            self.sequence_length,
            self.prediction_horizon,
            self.grid_size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
