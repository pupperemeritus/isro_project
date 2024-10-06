import polars as pl
import numpy as np
from darts import TimeSeries
from typing import Tuple
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
    return grid_lat, grid_lon


def interpolate_to_grid(points, values, grid_lat, grid_lon):
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
    interpolated = griddata(
        points, values, (grid_x, grid_y), method="linear", fill_value=np.nan
    )
    return interpolated.flatten()


def preprocess_data(
    df: pl.DataFrame,
    every: str,
    period: str,
    offset: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    grid_resolution: float,
) -> Tuple[TimeSeries, TimeSeries]:
    # Sort the DataFrame by time
    df = df.sort("IST_Time")

    # Create the grid aggregate function
    grid_agg = grid_aggregate(lat_range, lon_range, grid_resolution)

    # Group data by time windows and apply gridding using `pl.map_groups`
    grouped_data = df.group_by_dynamic(
        "IST_Time",
        every=every,
        period=period,
        offset=offset,
    ).agg(
        [
            pl.col("IST_Time").mean().alias("timestamp"),
            pl.map_groups(
                exprs=[
                    pl.col("Latitude").cast(pl.Float64),
                    pl.col("Longitude").cast(pl.Float64),
                    pl.col("Vertical S4").cast(pl.Float64),
                    pl.col("Vertical Scintillation Phase").cast(pl.Float64),
                ],
                function=lambda x: grid_agg(
                    pl.DataFrame(
                        {
                            "Latitude": x[0],
                            "Longitude": x[1],
                            "Vertical S4": x[2],
                            "Vertical Scintillation Phase": x[3],
                        }
                    )
                ),
            ).alias("gridded_data"),
        ]
    )

    

    # Extract the `gridded_data` fields using .arr.get() to access list-like data
    s4_data = grouped_data.select(pl.col("gridded_data").arr.get(0)).to_numpy()
    phase_data = grouped_data.select(pl.col("gridded_data").arr.get(1)).to_numpy()

    # Extract timestamps
    timestamps = grouped_data["timestamp"].to_numpy()

    # Create TimeSeries objects
    s4_series = TimeSeries.from_times_and_values(timestamps, np.stack(s4_data))
    phase_series = TimeSeries.from_times_and_values(timestamps, np.stack(phase_data))

    return s4_series, phase_series


def grid_aggregate(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    grid_resolution: float,
):
    grid_lat, grid_lon = create_grid(lat_range, lon_range, grid_resolution)

    def _grid_agg(group: pl.DataFrame):
        points = group.select(["Latitude", "Longitude"]).to_numpy()
        s4_values = group["Vertical S4"].to_numpy()
        phase_values = group["Vertical Scintillation Phase"].to_numpy()

        # Interpolate to grid and ensure consistency in output types
        s4_grid = np.array(
            interpolate_to_grid(points, s4_values, grid_lat, grid_lon), dtype=np.float32
        )
        phase_grid = np.array(
            interpolate_to_grid(points, phase_values, grid_lat, grid_lon),
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
