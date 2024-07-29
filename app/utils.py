import logging
import logging.config
from datetime import datetime, timedelta
from typing import List

import numpy as np
import polars as pl
from logging_conf import log_config

try:
    logging.config.dictConfig(log_config)
except Exception as e:
    logging.error(e)
logger = logging.Logger(__name__)


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
        "Vertical Scintillation Amplitude", amplitude_scintillation_slant / term_3
    )

    phase_scintillation_rad = df[
        "Phi60 on Sig1, 60-second phase sigma (radians)"
    ].to_numpy()

    term_4 = (1 / term_2) ** (0.5)
    vertical_scintillation_phase = pl.Series(
        "Vertical Scintillation Phase", phase_scintillation_rad / term_4
    )

    df.insert_column(-1, vertical_scintillation_amplitude)
    df.insert_column(-1, vertical_scintillation_phase)

    return df


def get_numeric_columns(df: pl.DataFrame) -> List[str]:
    logger.debug("Getting numeric columns")
    return df.select(pl.col(pl.FLOAT_DTYPES + pl.INTEGER_DTYPES)).columns


def get_categorical_columns(df: pl.DataFrame) -> List[str]:
    logger.debug("Getting categorical columns")
    return df.select(pl.col(pl.CATEGORICAL_DTYPES + pl.STRING_DTYPES)).columns


def calculate_single_lat_lon(elevation, azimuth, user_lat=17.39, user_long=78.31):
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


def add_lat_lon_to_df(df, elevation_col, azimuth_col, user_lat, user_lon):
    logger.info("Adding latitude and longitude to dataframe")
    lat, lon = calculate_lat_lon(
        df[elevation_col].to_numpy(), df[azimuth_col].to_numpy(), user_lat, user_lon
    )
    df = df.with_columns([pl.Series("Latitude", lat), pl.Series("Longitude", lon)])
    logger.debug("Latitude and Longitude columns added")
    return df


def gps_to_ist(gps_week, gps_seconds):
    logger.debug(f"Converting GPS time to IST: week={gps_week}, seconds={gps_seconds}")
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    gps_time = gps_epoch + timedelta(weeks=int(gps_week), seconds=float(gps_seconds))
    ist_time = gps_time + timedelta(hours=5,minutes=30)
    return ist_time


def calculate_lat_lon(elevation, azimuth, user_lat, user_long):
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


def process_csv_with_lat_lon(input_file, output_file, user_lat, user_lon):
    try:
        logger.info(f"Processing CSV file: {input_file}")
        df = pl.read_csv(input_file)
        df = add_lat_lon_to_df(df, "Elevation", "Azimuth", user_lat, user_lon)
        df.write_csv(output_file)
        logger.info(f"CSV file with latitude and longitude saved to {output_file}")
    except Exception as e:
        logger.exception(f"Error processing CSV file: {str(e)}")
