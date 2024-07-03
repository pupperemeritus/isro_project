import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def calculate_single_lat_lon(elevation, azimuth, user_lat, user_long):
    """
    Calculate latitude and longitude for a single point.
    """
    e_rad = elevation * (np.pi / 180)
    a_rad = azimuth * (np.pi / 180)
    user_lat_rad = user_lat * (np.pi / 180)
    user_long_rad = user_long * (np.pi / 180)

    earth_center_angle = (90 * (np.pi / 180)) - e_rad - np.arcsin(0.94 * np.cos(e_rad))
    earth_center_angle = max(0, earth_center_angle)

    ipp_lat_d1 = np.arcsin(
        (np.sin(user_lat_rad) * np.cos(earth_center_angle)) + (np.cos(user_lat_rad) * np.sin(earth_center_angle) * np.cos(a_rad))
    )
    ipp_lat = np.degrees(ipp_lat_d1)

    ipp_long_d3 = user_long_rad + np.arcsin((np.sin(earth_center_angle) * np.sin(a_rad)) / np.cos(ipp_lat_d1))
    ipp_lon = np.degrees(ipp_long_d3)

    return ipp_lat, ipp_lon


def calculate_lat_lon(elevation, azimuth, user_lat, user_long):
    """
    Calculate latitude and longitude for multiple points.
    """
    e_rad = np.radians(elevation)
    a_rad = np.radians(azimuth)
    u1_rad = np.radians(user_lat)
    u2_rad = np.radians(user_long)

    earth_center_angle = np.radians(90) - e_rad - np.arcsin(0.94 * np.cos(e_rad))
    earth_center_angle = np.maximum(0, earth_center_angle)

    d1 = np.arcsin(
        (np.sin(u1_rad) * np.cos(earth_center_angle)) + (np.cos(u1_rad) * np.sin(earth_center_angle) * np.cos(a_rad))
    )
    lat = np.degrees(d1)

    d3 = u2_rad + np.arcsin((np.sin(earth_center_angle) * np.sin(a_rad)) / np.cos(d1))
    lon = np.degrees(d3)

    return lat, lon


def add_lat_lon_to_df(df, elevation_col, azimuth_col, user_lat, user_lon):
    """
    Add latitude and longitude columns to a DataFrame.
    """
    df["Latitude"], df["Longitude"] = calculate_lat_lon(
        df[elevation_col], df[azimuth_col], user_lat, user_lon
    )
    return df


def gps_to_ist(gps_week, gps_seconds):
    """
    Converts GPS time of the week to IST (Indian Standard Time).
    """
    gps_epoch = datetime(1980, 1, 6, 5, 29, 42)
    gps_time_delta = timedelta(weeks=int(gps_week), seconds=float(gps_seconds))
    ist_time = gps_epoch + gps_time_delta
    return ist_time


def process_csv_with_lat_lon(input_file, output_file, user_lat, user_lon):
    """
    Process a CSV file, add latitude and longitude columns, and save to a new CSV.
    """
    df = pd.read_csv(input_file)
    df = add_lat_lon_to_df(
        df, "Elevation (degrees)", "Azimuth (degrees)", user_lat, user_lon
    )
    df.to_csv(output_file, index=False)
    print(f"CSV file with latitude and longitude saved to {output_file}")


# You can add the other utility functions here as well
