from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl
from darts import TimeSeries
from scipy.interpolate import griddata


def create_grid(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    grid_resolution: float,
):
    lat_start, lat_end = lat_range
    lon_start, lon_end = lon_range
    grid_lat = np.arange(lat_start, lat_end + grid_resolution, grid_resolution)
    grid_lon = np.arange(lon_start, lon_end + grid_resolution, grid_resolution)

    print("Created grid - Lat range:", grid_lat.min(), grid_lat.max())
    print("Created grid - Lon range:", grid_lon.min(), grid_lon.max())
    print("Grid shape:", grid_lat.shape, grid_lon.shape)

    return grid_lat, grid_lon


def preprocess_data(
    df: pl.DataFrame,
    every: str,
    period: str,
    offset: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    grid_resolution: float,
    error_margin: str = "5s",
) -> Tuple[TimeSeries, TimeSeries]:
    print("Initial data shape:", df.shape)
    print("NaN in initial data:", df.null_count().sum())

    # Sort the DataFrame by time
    df = df.sort("IST_Time")

    # Create the grid aggregate function
    grid_agg = grid_aggregate(lat_range, lon_range, grid_resolution)

    # Use custom grouping function with error margin
    grouped_data = custom_group_by_dynamic(
        df, "IST_Time", every, period, offset, error_margin
    )
    print("Number of groups:", len(grouped_data))

    # Apply grid aggregation
    results = apply_grid_aggregate(grouped_data, grid_agg)
    print("Number of results:", len(results))

    # Extract data from results
    timestamps = pd.DatetimeIndex([r["timestamp"] for r in results])
    s4_data = np.array([r["gridded_data"][0].flatten() for r in results])
    phase_data = np.array([r["gridded_data"][1].flatten() for r in results])

    print("S4 data shape:", s4_data.shape)
    print("Phase data shape:", phase_data.shape)
    print("NaN in S4 data:", np.isnan(s4_data).sum())
    print("NaN in Phase data:", np.isnan(phase_data).sum())

    # Create TimeSeries objects
    s4_series = TimeSeries.from_times_and_values(timestamps, s4_data)
    phase_series = TimeSeries.from_times_and_values(timestamps, phase_data)

    return s4_series, phase_series


def idw_interpolation(points, values, grid_lat, grid_lon, power=2, smoothing=1e-5):
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    grid_points = np.column_stack((grid_y.ravel(), grid_x.ravel()))

    # Remove NaN values from input data
    valid_indices = ~np.isnan(values)
    valid_points = points[valid_indices]
    valid_values = values[valid_indices]

    if len(valid_points) == 0:
        print("WARNING: No valid points for interpolation.")
        return np.zeros(grid_points.shape[0])

    # Calculate distances
    distances = np.sqrt(
        ((valid_points[:, None, :] - grid_points[None, :, :]) ** 2).sum(axis=2)
    )

    # Apply smoothing to avoid division by zero
    distances = np.maximum(distances, smoothing)

    # Calculate weights
    weights = 1.0 / (distances**power)

    # Normalize weights
    weights_sum = weights.sum(axis=0)
    weights /= weights_sum[None, :]

    # Interpolate
    interpolated = (weights * valid_values[:, None]).sum(axis=0)

    print("Interpolation result shape:", interpolated.shape)
    print(
        "Zero percentage after interpolation:",
        (interpolated == 0).sum() / interpolated.size * 100,
    )

    return interpolated


def direct_binning(points, values, grid_lat, grid_lon):
    lat_bins = np.digitize(points[:, 0], grid_lat)
    lon_bins = np.digitize(points[:, 1], grid_lon)

    grid = np.zeros((len(grid_lat), len(grid_lon)))
    for lat_bin, lon_bin, value in zip(lat_bins, lon_bins, values):
        grid[lat_bin - 1, lon_bin - 1] = (
            value  # -1 because digitize returns 1-indexed bins
        )

    return grid.flatten()


def average_time_windows(current_window, prev_window, next_window):
    windows = [w for w in [prev_window, current_window, next_window] if w is not None]
    return np.mean(windows, axis=0)


def interpolate_to_grid(
    points, values, grid_lat, grid_lon, prev_window=None, next_window=None
):
    print("Number of input points:", len(points))
    print("Input points range - Lat:", points[:, 0].min(), points[:, 0].max())
    print("Input points range - Lon:", points[:, 1].min(), points[:, 1].max())
    print("Grid Lat range:", grid_lat.min(), grid_lat.max())
    print("Grid Lon range:", grid_lon.min(), grid_lon.max())
    print("Number of non-NaN input values:", np.sum(~np.isnan(values)))

    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    # Remove NaN values from input data
    valid_indices = ~np.isnan(values)
    valid_points = points[valid_indices]
    valid_values = values[valid_indices]

    if len(valid_points) < 4:
        print(
            "WARNING: Not enough valid points for interpolation. Using direct binning."
        )
        return direct_binning(valid_points, valid_values, grid_lat, grid_lon)

    if len(valid_points) < 4:
        print("WARNING: Less than 4 valid points. Using average of time windows.")
        if prev_window is None or next_window is None:
            raise ValueError(
                "Previous and next window data required when current window has less than 4 points."
            )
        interpolated = average_time_windows(None, prev_window, next_window)
    else:
        # Perform IDW interpolation
        interpolated = idw_interpolation(valid_points, valid_values, grid_lat, grid_lon)

    interpolated = np.clip(interpolated, 0, None)

    return interpolated.flatten()


def custom_group_by_dynamic(
    df: pl.DataFrame,
    time_col: str,
    every: str,
    period: str,
    offset: str,
    error_margin: str = "5s",
):
    # Convert Polars DataFrame to pandas for easier datetime handling
    pdf = df.to_pandas()

    # Convert time strings to timedelta
    every_td = pd.Timedelta(every)
    period_td = pd.Timedelta(period)
    offset_td = pd.Timedelta(offset)
    error_margin_td = pd.Timedelta(error_margin)

    # Sort the dataframe by time
    pdf = pdf.sort_values(time_col)

    # Calculate the start and end times for the entire dataset
    start_time = pdf[time_col].min() - offset_td - error_margin_td
    end_time = pdf[time_col].max() + error_margin_td

    # Generate time windows
    windows = []
    current_time = start_time
    while current_time <= end_time:
        window_start = current_time - error_margin_td
        window_end = (
            window_start + period_td + (2 * error_margin_td)
        )  # Add margin to both start and end
        windows.append((window_start, window_end))
        current_time += every_td

    # Group data into windows
    grouped_data = []
    for window_start, window_end in windows:
        window_data = pdf[
            (pdf[time_col] >= window_start) & (pdf[time_col] < window_end)
        ]
        if not window_data.empty:
            grouped_data.append(
                {
                    "window_start": window_start
                    + error_margin_td,  # Adjust window start back to original
                    "window_end": window_end
                    - error_margin_td,  # Adjust window end back to original
                    "data": window_data,
                }
            )

    return grouped_data


def apply_grid_aggregate(grouped_data, grid_agg_func):
    results = []
    for i, group in enumerate(grouped_data):
        timestamp = (
            group["window_start"] + (group["window_end"] - group["window_start"]) / 2
        )
        df_group = pl.DataFrame(group["data"])
        prev_window = results[i - 1]["gridded_data"] if i > 0 else None
        next_window = grouped_data[i + 1]["data"] if i < len(grouped_data) - 1 else None
        if next_window is not None:
            next_window = pl.DataFrame(next_window)
            next_window = grid_agg_func(next_window, prev_window=None, next_window=None)
        gridded_data = grid_agg_func(
            df_group, prev_window=prev_window, next_window=next_window
        )
        results.append({"timestamp": timestamp, "gridded_data": gridded_data})
    return results


def grid_aggregate(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    grid_resolution: float,
):
    grid_lat, grid_lon = create_grid(lat_range, lon_range, grid_resolution)

    def _grid_agg(group: pl.DataFrame, prev_window=None, next_window=None):
        points = group.select(["Latitude", "Longitude"]).to_numpy()
        s4_values = group["Vertical S4"].to_numpy()
        phase_values = group["Vertical Scintillation Phase"].to_numpy()

        # Use robust interpolation
        s4_grid = np.array(
            interpolate_to_grid(
                points,
                s4_values,
                grid_lat,
                grid_lon,
                prev_window[0] if prev_window else None,
                next_window[0] if next_window else None,
            ),
            dtype=np.float32,
        )
        phase_grid = np.array(
            interpolate_to_grid(
                points,
                phase_values,
                grid_lat,
                grid_lon,
                prev_window[1] if prev_window else None,
                next_window[1] if next_window else None,
            ),
            dtype=np.float32,
        )

        # Ensure both grids are arrays of the same shape and type
        if s4_grid.shape != phase_grid.shape:
            raise ValueError("Inconsistent shapes between S4 and Phase grids.")

        return [s4_grid, phase_grid]

    return _grid_agg


def split_data(
    series: TimeSeries, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
    train_len = int(len(series) * train_frac)
    val_len = int(len(series) * val_frac)

    train_series = series[:train_len]
    val_series = series[train_len : train_len + val_len]
    test_series = series[train_len + val_len :]

    return train_series, val_series, test_series
