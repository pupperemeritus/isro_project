import logging
import logging.config
import os
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from logging_conf import log_config
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
            mapbox_layers=[
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [
                        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
                    ],
                }
            ],
        )
        logger.info(f"Map created successfully. Type: {map_type}")
        return fig
    except Exception as e:
        logger.exception(f"Error creating map: {str(e)}")
        return go.Figure()


def create_contour_map(
    df: pl.DataFrame,
    lat: str,
    lon: str,
    color: str,
    color_scale: str = "jet",
):
    df = df.drop_nulls(subset=[lat, lon, color])

    # Convert Polars Series to NumPy arrays
    points = np.array([df[lat].to_numpy(), df[lon].to_numpy()]).T
    values = df[color].to_numpy()
    values = np.nan_to_num(values, nan=0.0)

    # Define grid for interpolation
    grid_lat, grid_lon = np.mgrid[
        df[lat].min() : df[lat].max() : 100j,
        df[lon].min() : df[lon].max() : 100j,
    ]

    # Flatten the grid for comparison
    grid_points = np.array([grid_lat.flatten(), grid_lon.flatten()]).T

    # Find grid points that are not in the original data points
    unique_points = np.unique(points, axis=0)
    mask = np.all(np.isin(grid_points, unique_points))
    xi = grid_points[~mask].flatten()

    # Interpolate data onto the grid
    grid_values = griddata(
        points,
        values,
        (grid_lat, grid_lon),
        method="linear",
        rescale=True,
        fill_value=0.0,
    )

    # Create the plot
    fig, ax = plt.subplots(
        figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Add features to the map
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAKES, edgecolor="black")
    ax.add_feature(cfeature.RIVERS)

    # Add contour plot
    cont = ax.contourf(
        grid_lon[0, :],
        grid_lat[:, 0],
        grid_values,
        cmap=color_scale,
        levels=np.linspace(0, 1, 20),  # Ensure levels are within [0, 1]
        transform=ccrs.PlateCarree(),
    )
    cbar = fig.colorbar(cont, ax=ax, orientation="vertical", pad=0.1)
    cbar.set_label(color)  # Set colorbar label to reflect the color parameter

    # Set extent and title
    ax.set_extent(
        [df[lon].min(), df[lon].max(), df[lat].min(), df[lat].max()],
        crs=ccrs.PlateCarree(),
    )
    ax.set_title("Contour Map of " + color)

    return fig
