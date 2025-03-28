import logging
import logging.config
import os
import warnings
from datetime import datetime, timedelta
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

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


def create_grid(lon_range, lat_range, grid_resolution):
    lon_start, lon_end = lon_range
    lat_start, lat_end = lat_range

    lon_start, lon_end = lon_range
    lat_start, lat_end = lat_range

    grid_lon = np.arange(lon_start, lon_end + grid_resolution, grid_resolution)
    grid_lat = np.arange(lat_start, lat_end + grid_resolution, grid_resolution)

    return grid_lon, grid_lat


def interpolate_to_grid(points, values, grid_lon, grid_lat):
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    interpolated = griddata(
        points,
        values,
        (grid_x, grid_y),
        method="linear",
        fill_value=0,
    )
    return interpolated


class IonosphereDataset(Dataset):
    def __init__(
        self,
        dataframe: pl.DataFrame,
        time_window: str,
        sequence_length: int,
        prediction_horizon: int,
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        grid_resolution: float,
        stride: int,
    ):
        self.df = dataframe.sort("IST_Time")
        self.time_window = time_window
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride

        # Create fixed grid
        self.grid_lon, self.grid_lat = create_grid(
            lon_range, lat_range, grid_resolution
        )

        # Group data by time windows
        self.grouped_data = list(self.df.group_by_dynamic("IST_Time", every=time_window))

        # Create sequences
        self.sequences = [
            (i, i + sequence_length + prediction_horizon)
            for i in range(
                0,
                len(self.grouped_data) - sequence_length - prediction_horizon + 1,
                self.stride,
            )
        ]

        # Compute grid steps
        self.grid_lon_steps = len(self.grid_lon)
        self.grid_lat_steps = len(self.grid_lat)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start, end = self.sequences[idx]
        sequence_data = self.grouped_data[start : start + self.sequence_length]
        target_data = self.grouped_data[start + self.prediction_horizon : end]

        # Interpolate sequence data to grid
        sequence_s4 = np.stack(
            [
                interpolate_to_grid(
                    group[1][["Longitude", "Latitude"]].to_numpy(),
                    group[1]["Vertical S4"].to_numpy(),
                    self.grid_lon,
                    self.grid_lat,
                )
                for group in sequence_data
            ]
        )

        sequence_phase = np.stack(
            [
                interpolate_to_grid(
                    group[1][["Longitude", "Latitude"]].to_numpy(),
                    group[1]["Vertical Scintillation Phase"].to_numpy(),
                    self.grid_lon,
                    self.grid_lat,
                )
                for group in sequence_data
            ]
        )

        # Interpolate target data to grid
        target_s4 = np.stack(
            [
                interpolate_to_grid(
                    group[1][["Longitude", "Latitude"]].to_numpy(),
                    group[1]["Vertical S4"].to_numpy(),
                    self.grid_lon,
                    self.grid_lat,
                )
                for group in target_data
            ]
        )

        target_phase = np.stack(
            [
                interpolate_to_grid(
                    group[1][["Longitude", "Latitude"]].to_numpy(),
                    group[1]["Vertical Scintillation Phase"].to_numpy(),
                    self.grid_lon,
                    self.grid_lat,
                )
                for group in target_data
            ]
        )

        return {
            "features_s4": torch.from_numpy(
                sequence_s4
            ).float(),  # [seq_len, lat_steps, lon_steps]
            "features_phase": torch.from_numpy(
                sequence_phase
            ).float(),  # [seq_len, lat_steps, lon_steps]
            "target_s4": torch.from_numpy(
                target_s4
            ).float(),  # [pred_horizon, lat_steps, lon_steps]
            "target_phase": torch.from_numpy(
                target_phase
            ).float(),  # [pred_horizon, lat_steps, lon_steps]
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
        time_window: str,
        stride: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataframe"])
        self.dataframe = dataframe

        # Create grid
        self.grid_lon, self.grid_lat = create_grid(
            grid_lon_range, grid_lat_range, grid_resolution
        )

        # Compute grid steps
        self.grid_lon_steps = len(self.grid_lon)
        self.grid_lat_steps = len(self.grid_lat)

        # Calculate input size dynamically
        self.input_size = self.grid_lon_steps * self.grid_lat_steps * 2

    def setup(self, stage=None):
        train_df, val_df, test_df = self._split_data()

        dataset_params = dict(
            sequence_length=self.hparams.sequence_length,
            prediction_horizon=self.hparams.prediction_horizon,
            lon_range=self.hparams.grid_lon_range,
            lat_range=self.hparams.grid_lat_range,
            grid_resolution=self.hparams.grid_resolution,
            time_window=self.hparams.time_window,
            stride=self.hparams.stride,
        )

        self.train_dataset = IonosphereDataset(train_df, **dataset_params)
        self.val_dataset = IonosphereDataset(val_df, **dataset_params)
        self.test_dataset = IonosphereDataset(test_df, **dataset_params)

    def _split_data(self) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        total_samples = len(self.dataframe)
        train_size, val_size = int(0.7 * total_samples), int(0.15 * total_samples)

        return (
            self.dataframe.slice(0, train_size),
            self.dataframe.slice(train_size, train_size + val_size),
            self.dataframe.slice(train_size + val_size, total_samples),
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        def collate_fn(batch):
            collated = torch.utils.data.dataloader.default_collate(batch)
            return collated

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
