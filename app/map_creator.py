import logging
import logging.config
import os
from typing import Optional

from logging_conf import log_config
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.interpolate import griddata

try:
    logging.config.dictConfig(log_config)
except Exception as e:
    logging.error(e)
logger = logging.Logger(__name__)


def create_map(
    df: pl.DataFrame,
    lat: str,
    lon: str,
    color: str,
    size: Optional[str],
    map_type: str,
    map_style: str,
    zoom: float = 4.0,
    marker_size: int = 5,
    heatmap_size: int = 10,
    color_scale: Optional[str] = None,
    bin_heatmap: bool = False,
) -> go.Figure:
    try:
        logger.info(f"Creating {map_type} map")
        df = df.with_columns(
            [pl.col(lat).cast(pl.Float64), pl.col(lon).cast(pl.Float64)]
        ).drop_nulls(subset=[lat, lon])
        logger.debug(f"Data shape after cleaning: {df.shape}")

        center_lat = 20.593684
        center_lon = 78.96288

        if map_type == "Scatter":
            fig = px.scatter_mapbox(
                df.to_pandas(),
                lat=lat,
                lon=lon,
                color=color,
                size=size if size else None,
                height=1024,
                range_color=(0, 1),
                color_continuous_scale=color_scale if color_scale else None,
                hover_data=["SVID", "IST_Time", "S4"],
            )
            fig.update_traces(marker=dict(size=marker_size))
        elif map_type == "Heatmap":
            if bin_heatmap:
                # Define bin size and range
                bin_size = 1
                lat_range = (0, 40)
                lon_range = (65, 100)

                lat_bins = np.arange(lat_range[0], lat_range[1], bin_size)
                lon_bins = np.arange(lon_range[0], lon_range[1], bin_size)

                max_s4 = []
                for lat_val in lat_bins[:-1]:
                    for lon_val in lon_bins[:-1]:
                        mask = df["Latitude"].is_between(
                            lat_val, lat_val + bin_size
                        ) & df["Longitude"].is_between(lon_val, lon_val + bin_size)
                        if df.filter(mask).height > 0:
                            max_s4.append(df.filter(mask)["S4"].max())
                        else:
                            max_s4.append(np.nan)

                max_s4_df = pl.DataFrame(
                    {
                        "Latitude": np.repeat(lat_bins[:-1], len(lon_bins[:-1])),
                        "Longitude": np.tile(lon_bins[:-1], len(lat_bins[:-1])),
                        "S4": max_s4,
                    }
                )

                # Interpolate missing values
                x, y = np.meshgrid(lon_bins[:-1], lat_bins[:-1])
                z = max_s4_df["S4"].to_numpy()

                valid_idx = ~np.isnan(z)
                x_valid = x.flatten()[valid_idx]
                y_valid = y.flatten()[valid_idx]
                z_valid = z[valid_idx]

                interp_method = "cubic"
                interp_s4 = griddata(
                    (x_valid, y_valid), z_valid, (x, y), method=interp_method
                )

                max_s4_df = max_s4_df.with_columns(pl.Series("S4", interp_s4.flatten()))
                fig = px.density_mapbox(
                    max_s4_df.to_pandas(),
                    lat="Latitude",
                    lon="Longitude",
                    z="S4",
                    radius=heatmap_size,  # Adjust the radius as needed
                    range_color=(0, 1),  # Color range
                    color_continuous_scale=(
                        color_scale if color_scale else None
                    ),  # Color scale
                    title="Square Bin Heatmap of Max S4 Values",
                )
            else:
                fig = px.scatter_mapbox(
                    df.to_pandas(),
                    lat=lat,
                    lon=lon,
                    color=color,
                    size=color,  # Use S4 for both color and size
                    size_max=heatmap_size,
                    height=1024,
                    range_color=(0, 1),
                    color_continuous_scale=color_scale if color_scale else None,
                    hover_data=["SVID", "IST_Time", "S4"],
                )
        else:
            logger.error(f"Unsupported map type: {map_type}")
            return go.Figure()

        fig.update_layout(
            mapbox=dict(
                style=map_style,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom,  # Adjust this value to get the desired initial zoom level
            )
        )
        logger.info(f"Map created successfully. Type: {map_type}")
        return fig
    except Exception as e:
        logger.exception(f"Error creating map: {str(e)}")
        return go.Figure()
