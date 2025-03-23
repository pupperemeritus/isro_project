import logging
import logging.config
import os
import warnings
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import polars as pl
import streamlit as st
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from model.validation_functions import check_nan_and_inf  # Import
from model.validation_functions import (  # Import validation functions, not decorators
    validate_dataframe_columns,
    validate_column_data_types,
)

warnings.filterwarnings("ignore")

from model.logging_conf import get_logger  # Import get_logger for standardized logging
from model.grid_utils import create_grid, interpolate_to_grid  # NEW IMPORT

logger = get_logger(__name__)  # Initialize logger using get_logger()


def calculate_single_lat_lon(
    elevation: float, azimuth: float, user_lat: float = 17.39, user_lon: float = 78.31
):
    logger.debug(
        f"Calculating lat/lon for: el={elevation}, az={azimuth}, user_lat={user_lat}, user_lon={user_lon}"
    )
    e_rad = elevation * (np.pi / 180)
    a_rad = azimuth * (np.pi / 180)
    user_lat_rad = user_lat * (np.pi / 180)
    user_long_rad = user_lon * (np.pi / 180)

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
        df.select(elevation_col).to_numpy().flatten(),  # Use Polars select and flatten
        df.select(azimuth_col).to_numpy().flatten(),  # Use Polars select and flatten
        user_lat,
        user_lon,
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
    logger.info("Converting slant S4 and Phase to vertical...")
    try:
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
        logger.info("Slant to vertical conversion completed.")
        return df
    except Exception as e:
        logger.error(f"Error during slant to vertical conversion: {e}", exc_info=True)
        raise


def preprocess_dataframe(
    df: pl.DataFrame, user_lat: float, user_lon: float
) -> pl.DataFrame | None:
    logger.info("Preprocessing dataframe...")
    try:
        df = slant_to_vertical(df)
        df = add_lat_lon_to_df(df, "Elevation", "Azimuth", user_lat, user_lon)
        df = df.with_columns(
            [
                pl.struct(["GPS_WN", "GPS_TOW"])
                .map_elements(
                    lambda x: gps_to_ist(x["GPS_WN"], x["GPS_TOW"]),
                    return_dtype=datetime,
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

        if not valid_columns:
            logger.warning("No valid columns found after null percentage filtering.")
            return None  # Or raise an exception if valid columns are required

        # Select only valid columns
        df = df.select(valid_columns)
        logger.info("Dataframe preprocessing completed.")
        return df
    except Exception as e:
        logger.error(f"Error during dataframe preprocessing: {e}", exc_info=True)
        raise


def verify_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    required_columns = [
        "IST_Time",
        "Latitude",
        "Longitude",
        "Vertical S4",
        "Vertical Scintillation Phase",
    ]
    validate_dataframe_columns(df, required_columns)
    # Specify expected types (adjust as needed)
    column_types = {
        "Vertical S4": pl.Float64,
        "Vertical Scintillation Phase": pl.Float64,
    }
    validate_column_data_types(df, column_types)
    check_nan_and_inf(df, required_columns)
    return df


def load_data(file: str) -> Optional[pl.DataFrame]:
    logger.info(f"Starting to load data from file: {file}")
    try:
        if not os.path.exists(file):
            logger.error(f"Data file not found: {file}", exc_info=True)
            raise FileNotFoundError(f"Data file not found: {file}")

        if file.endswith(".arrow"):
            logger.info(f"Loading Arrow file: {file}")
            df = pl.read_ipc(file)
        elif file.endswith(".csv"):
            logger.info(f"Loading CSV file: {file}")
            df = pl.read_csv(file)
        elif file.endswith(".parquet"):
            logger.info(f"Loading Parquet file: {file}")
            df = pl.read_parquet(file)
        else:
            logger.error(
                "Unsupported file format. Supported formats are: .arrow, .csv, .parquet"
            )
            raise ValueError(
                "Unsupported file format. Supported formats are: .arrow, .csv, .parquet"
            )

        if df.is_empty():
            logger.warning("Loaded DataFrame is empty.")
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
            logger.error(
                f"Missing required columns: {', '.join(missing_columns)}", exc_info=True
            )
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        df = df.rename(
            {
                "WN, GPS Week Number": "GPS_WN",
                "TOW, GPS Time of Week (seconds)": "GPS_TOW",
                "Azimuth (degrees)": "Azimuth",
                "Elevation (degrees)": "Elevation",
                "Total S4 on Sig1 (dimensionless)": "S4",
            }
        )
        logger.debug(f"Columns after renaming: {df.columns}")
        logger.info(
            "Initial data loading and renaming completed. Proceeding to preprocess dataframe."
        )
        df = preprocess_dataframe(df, user_lat=17.39, user_lon=78.31)
        # NEW: verify dataframe integrity using validation functions
        if df is not None:
            df = verify_dataframe(df)
        validate_dataframe_columns(
            df,
            required_columns=[
                "IST_Time",
                "Latitude",
                "Longitude",
                "Vertical S4",
                "Vertical Scintillation Phase",
            ],
        )
        if df is None:  # preprocess_dataframe can return None if no valid columns
            logger.warning(
                "Preprocessing dataframe returned None (no valid columns). Data loading failed."
            )
            return None

        logger.info(f"Data loaded and preprocessed successfully. Shape: {df.shape}")
        return df

    except FileNotFoundError as e:
        logger.error(f"File not found error during data loading: {e}", exc_info=True)
        return None  # Handle FileNotFoundError specifically and return None

    except ValueError as e:
        logger.error(
            f"Value error during data loading or preprocessing: {e}", exc_info=True
        )
        return None  # Handle ValueError and return None

    except Exception as e:
        logger.exception(
            f"Unexpected error during data loading: {e}"
        )  # Use logger.exception for full traceback
        return None  # Catch any other exceptions and return None


class IonosphereDataset(Dataset):
    def __init__(
        self,
        dataframe: pl.DataFrame,
        time_window: str,
        sequence_length: int,
        prediction_horizon: int,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        grid_resolution: float,
        stride: int,
    ):
        # NEW: Verify dataframe in dataset initialization
        check_nan_and_inf(
            dataframe,
            [
                "IST_Time",
                "Latitude",
                "Longitude",
                "Vertical S4",
                "Vertical Scintillation Phase",
            ],
        )
        self.df = dataframe.sort("IST_Time")
        self.time_window = time_window
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride

        # Create fixed grid
        self.grid_lat, self.grid_lon = create_grid(
            lat_range,
            lon_range,
            grid_resolution,
        )

        # Group data by time windows
        self.grouped_data = list(
            self.df.group_by_dynamic("IST_Time", every=time_window)
        )

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
        self.grid_lat_steps = len(self.grid_lat)
        self.grid_lon_steps = len(self.grid_lon)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start, end = self.sequences[idx]
        sequence_data = self.grouped_data[start : start + self.sequence_length]
        target_data = self.grouped_data[start + self.sequence_length : end]

        # Ensure we have the correct number of time steps
        sequence_data = sequence_data[: self.sequence_length]
        target_data = target_data[: self.prediction_horizon]

        # Pad if necessary
        if len(sequence_data) < self.sequence_length:
            sequence_data = sequence_data + [sequence_data[-1]] * (
                self.sequence_length - len(sequence_data)
            )
        if len(target_data) < self.prediction_horizon:
            target_data = target_data + [target_data[-1]] * (
                self.prediction_horizon - len(target_data)
            )

        # Extract and convert timestamps
        def convert_timestamp(group):
            timestamp = group[1]["IST_Time"].mean()
            if isinstance(timestamp, datetime):
                return (timestamp - datetime(1970, 1, 1)).total_seconds()
            elif isinstance(timestamp, np.datetime64):
                return (
                    timestamp - np.datetime64("1970-01-01T00:00:00Z")
                ) / np.timedelta64(1, "s")
            else:
                raise ValueError(f"Unexpected timestamp type: {type(timestamp)}")

        sequence_timestamps = [convert_timestamp(group) for group in sequence_data]
        target_timestamps = [convert_timestamp(group) for group in target_data]

        # Interpolate sequence data to grid
        sequence_s4 = np.stack(
            [
                interpolate_to_grid(
                    group[1][["Latitude", "Longitude"]].to_numpy(),
                    group[1]["Vertical S4"].to_numpy(),
                    self.grid_lat,
                    self.grid_lon,
                )
                for group in sequence_data
            ]
        )
        sequence_phase = np.stack(
            [
                interpolate_to_grid(
                    group[1][["Latitude", "Longitude"]].to_numpy(),
                    group[1]["Vertical Scintillation Phase"].to_numpy(),
                    self.grid_lat,
                    self.grid_lon,
                )
                for group in sequence_data
            ]
        )

        # Interpolate target data to grid
        target_s4 = np.stack(
            [
                interpolate_to_grid(
                    group[1][["Latitude", "Longitude"]].to_numpy(),
                    group[1]["Vertical S4"].to_numpy(),
                    self.grid_lat,
                    self.grid_lon,
                )
                for group in target_data
            ]
        )
        target_phase = np.stack(
            [
                interpolate_to_grid(
                    group[1][["Latitude", "Longitude"]].to_numpy(),
                    group[1]["Vertical Scintillation Phase"].to_numpy(),
                    self.grid_lat,
                    self.grid_lon,
                )
                for group in target_data
            ]
        )

        return {
            "features_s4": torch.from_numpy(sequence_s4).float(),
            "features_phase": torch.from_numpy(sequence_phase).float(),
            "target_s4": torch.from_numpy(target_s4).float(),
            "target_phase": torch.from_numpy(target_phase).float(),
            "sequence_timestamps": torch.tensor(sequence_timestamps).float(),
            "target_timestamps": torch.tensor(target_timestamps).float(),
        }


class IonosphereDataModule(LightningDataModule):
    def __init__(
        self,
        dataframe: pl.DataFrame,
        sequence_length: int,
        prediction_horizon: int,
        grid_lat_range: Tuple[float, float],
        grid_lon_range: Tuple[float, float],
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
        self.grid_lat, self.grid_lon = create_grid(
            grid_lat_range, grid_lon_range, grid_resolution
        )

        # Compute grid steps
        self.grid_lat_steps = len(self.grid_lat)
        self.grid_lon_steps = len(self.grid_lon)

    def setup(self, stage=None):
        train_df, val_df, test_df = self._split_data()

        dataset_params = dict(
            sequence_length=self.hparams.sequence_length,
            prediction_horizon=self.hparams.prediction_horizon,
            lat_range=self.hparams.grid_lat_range,
            lon_range=self.hparams.grid_lon_range,
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
