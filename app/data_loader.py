import datetime
import hashlib
import logging
import logging.config
import time
from typing import Optional

import polars as pl
import streamlit as st
from logging_conf import log_config
from streamlit.runtime.uploaded_file_manager import UploadedFile
from utils import add_lat_lon_to_df, gps_to_ist

try:
    logging.config.dictConfig(log_config)
except Exception as e:
    logging.error(e)
logger = logging.Logger(__name__)


@st.cache_resource
def load_data(file: UploadedFile) -> Optional[pl.DataFrame]:
    try:
        logger.info(f"Starting to load data from file: {file.name}")
        if file.name.endswith(".arrow"):
            # Polars uses read_ipc for Arrow IPC format instead of read_csv
            df = pl.read_ipc(file)
        elif file.name.endswith(".csv"):
            df = pl.read_csv(file)

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
                    return_dtype=datetime.datetime,
                )
                .alias("UTC_Time")
            ]
        )
        # Polars doesn't have a direct equivalent to pd.Timedelta, use datetime
        df = df.with_columns(
            [(pl.col("UTC_Time") + pl.duration(hours=5, minutes=30)).alias("IST_Time")]
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


class FileWatcher:
    def __init__(self, file, callback):
        self.file = file
        self.callback = callback
        self.previous_hash = None
        self.is_running = True

    def watch(self):
        logger.info("File watcher started")
        while self.is_running:
            current_hash = hashlib.md5(self.file.getvalue()).hexdigest()
            if current_hash != self.previous_hash:
                logger.info("File change detected. Reloading data.")
                self.previous_hash = current_hash
                df = load_data(self.file)
                if df is not None:
                    self.callback(df)
            time.sleep(100)

    def stop(self):
        logger.info("Stopping file watcher")
        self.is_running = False


def create_file_watcher(file, callback):
    logger.info("Creating file watcher")
    return FileWatcher(file, callback)
