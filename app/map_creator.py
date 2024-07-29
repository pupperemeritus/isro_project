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
    zoom: float = 3.6,
    marker_size: int = 5,
    heatmap_size: int = 20,
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
        match map_type:
            case "Scatter":
                fig = px.scatter_mapbox(
                    df.to_pandas(),
                    lat=lat,
                    lon=lon,
                    color=color,
                    size=size if size else None,
                    height=1024,
                    range_color=(0, 1),
                    color_continuous_scale=color_scale if color_scale else None,
                    hover_data=[
                        "SVID",
                        "IST_Time",
                        "S4",
                        "Vertical Scintillation Amplitude",
                    ],
                    title="Scatter Mapbox for S4 Values",
                )
                fig.update_traces(marker=dict(size=marker_size))
            case "Heatmap":
                if bin_heatmap:
                    # Define bin size and range
                    bin_size = 1
                    lat_range = (0, 40)
                    lon_range = (65, 100)

                    lat_bins = np.arange(lat_range[0], lat_range[1], bin_size)
                    lon_bins = np.arange(lon_range[0], lon_range[1], bin_size)

                    # Create a 2D grid for the binned data
                    binned_data = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))

                    for i, lat_val in enumerate(lat_bins[:-1]):
                        for j, lon_val in enumerate(lon_bins[:-1]):
                            mask = df["Latitude"].is_between(
                                lat_val, lat_val + bin_size
                            ) & df["Longitude"].is_between(lon_val, lon_val + bin_size)
                            filtered = df.filter(mask)
                            if filtered.height > 0:
                                binned_data[i, j] = filtered[
                                    "Vertical Scintillation Amplitude"
                                ].max()

                    # Create meshgrid for interpolation
                    x, y = np.meshgrid(
                        lon_bins[:-1] + bin_size / 2, lat_bins[:-1] + bin_size / 2
                    )

                    # Mask out empty bins
                    valid_mask = ~np.isnan(binned_data)
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    z_valid = binned_data[valid_mask]

                    # Interpolate
                    grid_x, grid_y = np.mgrid[
                        lon_range[0] : lon_range[1] : 100j,
                        lat_range[0] : lat_range[1] : 100j,
                    ]
                    grid_z = griddata(
                        (x_valid, y_valid),
                        z_valid,
                        (grid_x, grid_y),
                        method="cubic",
                        fill_value=0,
                    )

                    # Create the density_mapbox
                    fig = go.Figure(
                        go.Densitymapbox(
                            lat=grid_y.flatten(),
                            lon=grid_x.flatten(),
                            z=grid_z.flatten(),
                            radius=20,
                            colorscale=color_scale if color_scale else "Viridis",
                            zmin=0,
                            zmax=1,
                            colorbar=dict(title="Vertical Scintillation Amplitude"),
                            hovertext=[
                                f"Vertical S4: {s4:.2f}" for s4 in grid_z.flatten()
                            ],
                            hoverinfo="text+lon+lat",
                            hovertemplate="Vertical S4: %{z:.2f}\nLatitude: %{lat:.0f}\nLong: %{lon:.0f}<extra></extra>",
                        )
                    )

                    # Update the layout
                    fig.update_layout(
                        title="Intensity-based Heatmap of S4 Values",
                        mapbox_style=map_style,
                        mapbox=dict(
                            center=dict(lat=20, lon=82.5), zoom=4
                        ),  # Center of India
                        width=1000,
                        height=800,
                    )
                else:
                    fig = px.scatter_mapbox(
                        df.to_pandas(),
                        lat=lat,
                        lon=lon,
                        color=color,
                        size=df["Vertical Scintillation Amplitude"]
                        .fill_nan(0)
                        .to_numpy()
                        / 2,  # Use S4 for both color and size
                        size_max=heatmap_size,
                        height=1024,
                        title="Scatter Mapbox for S4 Values",
                        range_color=(0, 1),
                        color_continuous_scale=color_scale if color_scale else None,
                        hover_data=[
                            "SVID",
                            "IST_Time",
                            "Vertical Scintillation Amplitude",
                        ],
                    )
            case "":
                logger.error(f"Unsupported map type: {map_type}")
                return go.Figure()

        fig.update_layout(
            mapbox=dict(
                style=map_style,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom,  # Adjust this value to get the desired initial zoom level
            ),
        )
        logger.info(f"Map created successfully. Type: {map_type}")
        return fig
    except Exception as e:
        logger.exception(f"Error creating map: {str(e)}")
        return go.Figure()
