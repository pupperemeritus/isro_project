import numpy as np
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
import polars as pl
import logging

logger = logging.getLogger(__name__)


def create_grid(lat_range, lon_range, grid_resolution):
    lat_start, lat_end = lat_range
    lon_start, lon_end = lon_range
    grid_lat = np.arange(lat_start, lat_end + grid_resolution, grid_resolution)
    grid_lon = np.arange(lon_start, lon_end + grid_resolution, grid_resolution)
    return grid_lat, grid_lon


def direct_binning(points, values, grid_lat, grid_lon):
    lat_min, lat_max = np.min(grid_lat), np.max(grid_lat)
    lon_min, lon_max = np.min(grid_lon), np.max(grid_lon)
    binned, _, _, _ = binned_statistic_2d(
        points[:, 1],  # Longitude
        points[:, 0],  # Latitude
        values,
        bins=[len(grid_lon), len(grid_lat)],
        statistic="mean",
        range=[[lon_min, lon_max], [lat_min, lat_max]],
    )
    # Transpose and replace NaNs
    binned = np.nan_to_num(binned.T, nan=0)
    return binned.flatten()


def interpolate_to_grid(points, values, grid_lat, grid_lon, min_points=4):
    if len(points) < min_points:
        logger.debug("Not enough points for interpolation; using direct binning.")
        return direct_binning(points, values, grid_lat, grid_lon)
    try:
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
        interpolated = griddata(
            points, values, (grid_x, grid_y), method="linear", fill_value=np.nan
        )
        # Fallback to binning if necessary
        if np.all(np.isnan(interpolated)):
            logger.debug("Interpolation resulted in all NaNs; falling back to binning.")
            return direct_binning(points, values, grid_lat, grid_lon)
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            binned = direct_binning(points, values, grid_lat, grid_lon)
            interpolated[nan_mask] = binned[nan_mask]
        return interpolated.flatten()
    except Exception as e:
        logger.error(f"Error during interpolation: {e}")
        return direct_binning(points, values, grid_lat, grid_lon)
