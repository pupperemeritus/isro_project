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
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.spatial import cKDTree

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
            case "Scatter/Heatmap":
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
                                binned_data[i, j] = filtered["Vertical S4"].max()

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
                            colorbar=dict(title="Vertical S4"),
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
                        mapbox=dict(center=dict(lat=20, lon=82.5), zoom=4),
                    )
                else:
                    fig = px.scatter_mapbox(
                        df.to_pandas(),
                        lat=lat,
                        lon=lon,
                        color=color,
                        size=df["Vertical S4"].fill_nan(0).to_numpy()
                        / 2,  # Use S4 for both color and size
                        size_max=heatmap_size,
                        height=1024,
                        title="Scatter Mapbox for S4 Values",
                        range_color=(0, 1),
                        color_continuous_scale=color_scale if color_scale else None,
                        hover_data=[
                            "SVID",
                            "IST_Time",
                            "Vertical S4",
                        ],
                    )
                    fig.update_layout(
                        mapbox_style=map_style,
                        mapbox=dict(center=dict(lat=20, lon=82.5), zoom=4),
                    )
            case _:
                logger.error(f"Unsupported map type: {map_type}", exc_info=True)
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


def idw_interpolation(x, y, z, xi, yi, power=4):
    dist = np.sqrt((x[:, np.newaxis] - xi) ** 2 + (y[:, np.newaxis] - yi) ** 2)
    weights = 1.0 / (dist**power + 1e-8)  # Add small constant to avoid division by zero
    zi = np.sum(weights * z[:, np.newaxis], axis=0) / np.sum(weights, axis=0)
    return zi


def create_contour_map(
    df: pl.DataFrame,
    lat: str,
    lon: str,
    color: str,
    color_scale: str = "jet",
    n_neighbors: int = 5,
    power: float = 4,
):
    df = df.drop_nulls(subset=[lat, lon, color])

    # Convert Polars Series to NumPy arrays
    points = np.array([df[lat].to_numpy(), df[lon].to_numpy()]).T
    values = df[color].to_numpy()

    grid_limits = [
        min((df["Latitude"].min() - df["Latitude"].min() % 5), 0),
        max((df["Latitude"].max() + (5 - df["Latitude"].max() % 5)), 40),
        min((df["Longitude"].min() - df["Longitude"].min() % 5), 65),
        max((df["Longitude"].max() + (5 - df["Longitude"].max() % 5)), 100),
    ]
    x_res = (grid_limits[1] - grid_limits[0]) * 1j
    y_res = (grid_limits[3] - grid_limits[2]) * 1j
    # Define grid for interpolation
    grid_lat, grid_lon = np.mgrid[
        grid_limits[0] : grid_limits[1] : x_res,
        grid_limits[2] : grid_limits[3] : y_res,
    ]

    # Flatten the grid points
    grid_points = np.column_stack((grid_lat.ravel(), grid_lon.ravel()))

    # Build a KD-tree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Find the nearest neighbors for each grid point
    distances, indices = tree.query(grid_points, k=n_neighbors)

    # Calculate IDW weights
    weights = 1.0 / (distances**power + 1e-8)
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    # Interpolate values
    grid_values = np.sum(weights * values[indices], axis=1)
    grid_values = grid_values.reshape(grid_lat.shape)

    grid_values = np.nan_to_num(grid_values, nan=0)

    # Create the plot
    fig, ax = plt.subplots(
        figsize=(16, 9),
        subplot_kw={"projection": ccrs.PlateCarree()},
        sharex=True,
        sharey=True,
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
        levels=np.linspace(0, 1, 500),  # Ensure levels are within [0, 1]
        transform=ccrs.PlateCarree(),
    )
    ax.set_extent(
        [grid_limits[2], grid_limits[3], grid_limits[0], grid_limits[1]],
        crs=ccrs.PlateCarree(),
    )
    ax.set_xticks(
        np.arange(grid_limits[2], grid_limits[3] + 1, 5), crs=ccrs.PlateCarree()
    )
    ax.set_yticks(
        np.arange(grid_limits[0], grid_limits[1] + 1, 5), crs=ccrs.PlateCarree()
    )
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    cbar = fig.colorbar(cont, ax=ax, orientation="vertical", pad=0.1)
    cbar.set_label(color)  # Set colorbar label to reflect the color parameter

    # Set extent and title
    ax.set_title("Contour Map of " + color)

    return fig
