import numpy as np
import pandas as pd
import polars as pl
from typing import Tuple, Callable, Optional, List, Any
from darts import TimeSeries
from numpy import floating
from numpy.typing import NDArray
from typing_extensions import TypeAlias
from scipy.interpolate import griddata
from model.logging_conf import get_logger
from model.grid_utils import create_grid, interpolate_to_grid
from model.validation_functions import check_nan_and_inf

GridAggFunction: TypeAlias = Callable[
    [pl.DataFrame, Optional[NDArray[np.float32]], Optional[NDArray[np.float32]]],
    List[NDArray[np.float32]],
]
logger = get_logger(__name__)


def preprocess_data(
    df: pl.DataFrame,
    every: str,
    period: str,
    offset: str,
    lat_range: Tuple[floating[Any], floating[Any]],
    lon_range: Tuple[floating[Any], floating[Any]],
    grid_resolution: floating[Any],
    error_margin: str = "5s",
) -> Tuple[TimeSeries, TimeSeries]:  # Return only two TimeSeries now
    logger.info("Preprocessing time series data using Polars...")
    logger.debug(
        f"Preprocessing parameters: every={every}, period={period}, offset={offset}, lat_range={lat_range}, lon_range={lon_range}, grid_resolution={grid_resolution}, error_margin={error_margin}"
    )
    try:
        logger.info("Sorting DataFrame by IST_Time using Polars...")
        df = df.sort("IST_Time")

        logger.info("Creating grid aggregate function...")
        grid_agg = grid_aggregate(lat_range, lon_range, grid_resolution)

        logger.info("Grouping data dynamically using Polars-compatible method...")
        grouped_data = custom_group_by_dynamic(
            df, "IST_Time", every, period, offset, error_margin
        )
        logger.info(f"Data grouped into {len(grouped_data)} time windows.")

        logger.info("Applying grid aggregation...")
        results = apply_grid_aggregate(grouped_data, grid_agg)
        logger.info(f"Grid aggregation applied to {len(results)} time windows.")

        logger.info("Extracting data from results...")
        timestamps = pd.DatetimeIndex([r["timestamp"] for r in results])
        s4_data = np.array([r["gridded_data"][0].flatten() for r in results])
        phase_data = np.array([r["gridded_data"][1].flatten() for r in results])
        logger.debug(
            f"Extracted data shapes: S4 data: {s4_data.shape}, Phase data: {phase_data.shape}"
        )

        # Data Statistics Logging
        logger.info(
            f"S4 Data - Min: {np.nanmin(s4_data)}, Max: {np.nanmax(s4_data)}, Mean: {np.nanmean(s4_data)}, Std: {np.nanstd(s4_data)}, NaN count: {np.isnan(s4_data).sum()}"
        )
        logger.info(
            f"Phase Data - Min: {np.nanmin(phase_data)}, Max: {np.nanmax(phase_data)}, Mean: {np.nanmean(phase_data)}, Std: {np.nanstd(phase_data)}, NaN count: {np.isnan(phase_data).sum()}"
        )

        logger.info("Creating TimeSeries objects...")
        s4_series = TimeSeries.from_times_and_values(timestamps, s4_data)
        phase_series = TimeSeries.from_times_and_values(timestamps, phase_data)
        logger.info("TimeSeries objects created successfully.")

        logger.info("Preprocessing of time series data completed.")
        return s4_series, phase_series  # Return only two TimeSeries
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}", exc_info=True)
        raise


def custom_group_by_dynamic(
    df: pl.DataFrame,
    time_col: str,
    every: str,
    period: str,
    offset: str,
    error_margin: str = "5s",
):
    logger.info("Grouping data by dynamic time windows...")
    logger.debug(
        f"Grouping parameters: time_col={time_col}, every={every}, period={period}, offset={offset}, error_margin={error_margin}"
    )
    try:
        pdf = df.to_pandas()
        every_td = pd.Timedelta(every)
        period_td = pd.Timedelta(period)
        offset_td = pd.Timedelta(offset)
        error_margin_td = pd.Timedelta(error_margin)

        pdf = pdf.sort_values(time_col)

        # Temporal Consistency Check - within group_by_dynamic
        time_diffs_grouping = pdf[time_col].diff().dropna()
        if not (time_diffs_grouping.min() >= pd.Timedelta(0)):
            logger.warning(
                "Potential temporal issue within time window grouping: Timestamps are not strictly increasing within group."
            )

        start_time = pdf[time_col].min() - offset_td - error_margin_td
        end_time = pdf[time_col].max() + error_margin_td

        windows = []
        current_time = start_time
        while current_time <= end_time:
            window_start = current_time - error_margin_td
            window_end = window_start + period_td + (2 * error_margin_td)
            windows.append((window_start, window_end))
            current_time += every_td

        logger.debug(f"Generated {len(windows)} time windows.")

        grouped_data = []
        for window_start, window_end in windows:
            window_data = pdf[
                (pdf[time_col] >= window_start) & (pdf[time_col] < window_end)
            ]
            if not window_data.empty:
                grouped_data.append(
                    {
                        "window_start": window_start + error_margin_td,
                        "window_end": window_end - error_margin_td,
                        "data": window_data,
                    }
                )
        logger.info(f"Data grouped into {len(grouped_data)} dynamic time windows.")
        return grouped_data
    except Exception as e:
        logger.error(f"Error during dynamic time window grouping: {e}", exc_info=True)
        raise


def apply_grid_aggregate(grouped_data, grid_agg_func: GridAggFunction):
    logger.debug("Applying grid aggregation to grouped data...")
    logger.debug(f"Number of groups to aggregate: {len(grouped_data)}")
    results = []
    try:
        for i, group in enumerate(grouped_data):
            timestamp = (
                group["window_start"]
                + (group["window_end"] - group["window_start"]) / 2
            )
            df_group = pl.DataFrame(group["data"])
            prev_window = results[i - 1]["gridded_data"] if i > 0 else None
            next_window = (
                grouped_data[i + 1]["data"] if i < len(grouped_data) - 1 else None
            )
            if next_window is not None:
                next_window = pl.DataFrame(next_window)
                next_window = grid_agg_func(
                    next_window, prev_window=None, next_window=None
                )  # Aggregate next window for interpolation
            gridded_data = grid_agg_func(
                df_group, prev_window=prev_window, next_window=next_window
            )
            results.append({"timestamp": timestamp, "gridded_data": gridded_data})
        logger.debug("Grid aggregation applied successfully to all groups.")
        return results
    except Exception as e:
        logger.error(f"Error applying grid aggregation: {e}", exc_info=True)
        raise


def grid_aggregate(
    lat_range: Tuple[floating[Any], floating[Any]],
    lon_range: Tuple[floating[Any], floating[Any]],
    grid_resolution: floating[Any],
):
    logger.debug("Creating grid aggregate function (grid_aggregate)...")
    grid_lat, grid_lon = create_grid(lat_range, lon_range, grid_resolution)

    def _grid_agg(group: pl.DataFrame, prev_window=None, next_window=None):
        points = group.select(["Latitude", "Longitude"]).to_numpy()
        s4_values = group.select("Vertical S4").to_numpy().flatten()
        phase_values = group.select("Vertical Scintillation Phase").to_numpy().flatten()
        logger.debug(
            f"grid_aggregate._grid_agg - s4_values.shape: {s4_values.shape}, phase_values.shape: {phase_values.shape}"
        )
        s4_grid = np.array(
            interpolate_to_grid(
                points,
                s4_values,
                grid_lat,
                grid_lon,
            ),
            dtype=np.float32,
        )
        phase_grid = np.array(
            interpolate_to_grid(
                points,
                phase_values,
                grid_lat,
                grid_lon,
            ),
            dtype=np.float32,
        )
        if s4_grid.shape != phase_grid.shape:
            error_msg = (
                "Inconsistent shapes between S4 and Phase grids after interpolation."
            )
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg)
        return [s4_grid, phase_grid]

    logger.debug("Grid aggregate function created.")
    return _grid_agg


def split_data(
    series: TimeSeries, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
    logger.debug("Splitting data into train, validation, and test sets...")
    logger.debug(
        f"Split parameters: train_frac={train_frac}, val_frac={val_frac}, series length={len(series)}"
    )
    try:
        train_len = int(len(series) * train_frac)
        val_len = int(len(series) * val_frac)

        train_series = series[:train_len]
        val_series = series[train_len : train_len + val_len]
        test_series = series[train_len + val_len :]

        logger.debug(
            f"Data split completed. Train size: {len(train_series)}, Validation size: {len(val_series)}, Test size: {len(test_series)}"
        )
        return train_series, val_series, test_series
    except Exception as e:
        logger.error(f"Error splitting data: {e}", exc_info=True)
        raise
