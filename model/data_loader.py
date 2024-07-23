import hashlib
import logging
import logging.config
import time
import warnings
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import polars as pl
import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interpolate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


logger = logging.Logger(__name__)


def get_numeric_columns(df: pl.DataFrame) -> List[str]:
    logger.debug("Getting numeric columns")
    return df.select(pl.col(pl.FLOAT_DTYPES + pl.INTEGER_DTYPES)).columns


def get_categorical_columns(df: pl.DataFrame) -> List[str]:
    logger.debug("Getting categorical columns")
    return df.select(pl.col(pl.CATEGORICAL_DTYPES + pl.STRING_DTYPES)).columns


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


def slant_to_vertical_s4(df: pl.DataFrame):
    e_rad = df["Elevation"].to_numpy()
    radius_earth_km = 6378
    height_ipp_km = 350

    p = (
        df[
            "p on Sig1, spectral slope of detrended phase in the 0.1 to 25Hz range (dimensionless)"
        ]
        .cast(pl.Float64)
        .drop_nulls()
        .to_numpy()
    )

    amplitude_scintillation_slant = df["Total S4 on Sig1 (dimensionless)"].to_numpy()

    term_1 = radius_earth_km * np.cos(e_rad) / (radius_earth_km + height_ipp_km)
    term_2 = np.sqrt(1 - term_1**2)
    term_3 = (1 / term_2) ** (p + 0.25)
    vertical_scintillation_amplitude = amplitude_scintillation_slant / term_3

    phase_scintillation_rad = df[
        "Phi60 on Sig1, 60-second phase sigma (radians)"
    ].to_numpy()

    term_4 = (1 / term_2) ** (0.5)
    vertical_scintillation_phase = phase_scintillation_rad / term_4

    df["Vertical Scintillation Amplitude"] = vertical_scintillation_amplitude
    df["Vertical Scintillation Phase"] = vertical_scintillation_phase


def load_data(file: str) -> Optional[pl.DataFrame]:
    try:
        logger.info(f"Starting to load data from file: {file}")
        if file.endswith(".arrow"):
            # Polars uses read_ipc for Arrow IPC format instead of read_csv
            df = pl.read_ipc(file)
        elif file.endswith(".csv"):
            df = pl.read_csv(file)
        elif file.endswith(".parquet"):
            df = pl.read_parquet(file)

        if df.is_empty():  # Polars uses is_empty() instead of empty attribute
            logger.warning("Loaded empty DataFrame from IPC file")

        required_columns = [
            "WN, GPS Week Number",
            "TOW, GPS Time of Week (seconds)",
            "SVID",
            "Azimuth (degrees)",
            "Elevation (degrees)",
            "Total S4 on Sig1 (dimensionless)",
        ]

        # Polars uses a different method to check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None

        logger.debug("Renaming columns for easier handling")
        # Polars uses a different syntax for renaming columns
        df = df.rename(
            {
                "WN, GPS Week Number": "GPS_WN",
                "TOW, GPS Time of Week (seconds)": "GPS_TOW",
                "Azimuth (degrees)": "Azimuth",
                "Elevation (degrees)": "Elevation",
                "Total S4 on Sig1 (dimensionless)": "S4",
            }
        )

        logger.debug("Adding latitude and longitude columns")
        user_lat = 17.39
        user_lon = 78.31
        # Assuming add_lat_lon_to_df is adapted for Polars
        df = add_lat_lon_to_df(df, "Elevation", "Azimuth", user_lat, user_lon)

        logger.debug("Converting GPS time to UTC and IST")
        # Polars uses a different approach for applying functions to columns
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

        logger.debug("Performing basic data cleaning")
        # Polars uses drop_nulls instead of dropna
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
        )
        df = df.with_columns(pl.col("IST_Time").cast(pl.Datetime))

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Error loading data: {str(e)}")
        return None


class IonosphereDataset(Dataset):
    def __init__(
        self,
        dataframe,
        target_column="S4",
        sequence_length=60,
        prediction_horizon=1,
        missing_data="closest",
        max_gap=5,
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.missing_data = missing_data
        self.max_gap = max_gap

        # Sort the dataframe by time
        dataframe = dataframe.sort("IST_Time")

        # Use your existing function to get numeric columns
        feature_columns = get_numeric_columns(dataframe)
        feature_columns = [col for col in feature_columns if col != target_column]

        # Normalize features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(dataframe[feature_columns].to_numpy())

        # Prepare target
        self.target = dataframe[target_column].to_numpy()

        # Get timestamps
        self.timestamps = dataframe["IST_Time"].to_numpy()

    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        start_time = self.timestamps[idx]
        end_time = start_time + np.timedelta64(self.sequence_length - 1, "m")
        target_time = end_time + np.timedelta64(self.prediction_horizon, "m")

        # Find indices of data points within the time range
        time_mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)
        target_idx = np.searchsorted(self.timestamps, target_time)

        if self.missing_data == "closest":
            # Get the available data points
            available_times = self.timestamps[time_mask]
            available_features = self.features[time_mask]

            # Create a sequence of minute-by-minute timestamps
            full_sequence_times = np.arange(
                start_time, end_time + np.timedelta64(1, "m"), np.timedelta64(1, "m")
            )

            # Find the closest available data point for each minute
            closest_indices = np.searchsorted(available_times, full_sequence_times)
            closest_indices = np.clip(closest_indices, 0, len(available_times) - 1)

            # Create the feature sequence using the closest available data
            feature_seq = available_features[closest_indices]

        elif self.missing_data == "interpolate":
            # Create a sequence of minute-by-minute timestamps
            full_sequence_times = np.arange(
                start_time, end_time + np.timedelta64(1, "m"), np.timedelta64(1, "m")
            )

            # Convert timestamps to numbers for interpolation
            available_times_num = (
                self.timestamps[time_mask] - start_time
            ) / np.timedelta64(1, "m")
            full_sequence_times_num = (
                full_sequence_times - start_time
            ) / np.timedelta64(1, "m")

            # Interpolate each feature
            feature_seq = np.zeros((len(full_sequence_times), self.features.shape[1]))
            for i in range(self.features.shape[1]):
                f = interpolate.interp1d(
                    available_times_num,
                    self.features[time_mask, i],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                feature_seq[:, i] = f(full_sequence_times_num)

        # Check if the gap is too large
        if np.any(np.diff(full_sequence_times) > np.timedelta64(self.max_gap, "m")):
            return None

        feature_seq = torch.FloatTensor(feature_seq)
        target = torch.FloatTensor([self.target[target_idx]])
        return feature_seq, target


def prepare_data_loaders(
    file_path,
    batch_size=32,
    sequence_length=60,
    prediction_horizon=1,
    test_size=0.2,
    val_size=0.1,
    missing_data="closest",
    max_gap=5,
):
    # Load the data using your existing function
    df = load_data(file_path)

    if df is None:
        raise ValueError("Failed to load data")

    # Convert Polars DataFrame to Pandas for train_test_split
    df_pandas = df.to_pandas()

    # Split the data
    train_val_df, test_df = train_test_split(
        df_pandas, test_size=test_size, random_state=42
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size / (1 - test_size), random_state=42
    )

    # Convert back to Polars DataFrames
    train_df = pl.DataFrame(train_df)
    val_df = pl.DataFrame(val_df)
    test_df = pl.DataFrame(test_df)

    # Create datasets
    train_dataset = IonosphereDataset(
        train_df,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        missing_data=missing_data,
        max_gap=max_gap,
    )
    val_dataset = IonosphereDataset(
        val_df,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        missing_data=missing_data,
        max_gap=max_gap,
    )
    test_dataset = IonosphereDataset(
        test_df,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        missing_data=missing_data,
        max_gap=max_gap,
    )

    # Create data loaders with a custom collate function to handle None values
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
