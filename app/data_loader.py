import hashlib
import logging
import logging.config
import os
import time
from typing import Optional

import pandas as pd
from utils import add_lat_lon_to_df, gps_to_ist

try:
    logging.config.fileConfig(
        os.path.join(os.getcwd(), "app", "logging.conf"), disable_existing_loggers=False
    )
except Exception as e:
    logging.error("Cwd must be root of project directory")
logger = logging.Logger(__name__)


def load_data(file) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Starting to load data from file: {file.name}")
        df = pd.read_csv(file)

        if df.empty:
            logger.warning("Loaded empty DataFrame from CSV")

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

        logger.debug("Renaming columns for easier handling")
        df = df.rename(
            columns={
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
        df = add_lat_lon_to_df(df, "Elevation", "Azimuth", user_lat, user_lon)

        logger.debug("Converting GPS time to UTC and IST")
        df["UTC_Time"] = df.apply(
            lambda row: gps_to_ist(row["GPS_WN"], row["GPS_TOW"]), axis=1
        )
        df["IST_Time"] = df["UTC_Time"] + pd.Timedelta(hours=5, minutes=30)

        logger.debug("Performing basic data cleaning")
        df = df.dropna(
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
