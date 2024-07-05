import logging
import logging.config
import os
from typing import Optional
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    logging.config.fileConfig(
        os.path.join(os.getcwd(), "app", "logging.conf"), disable_existing_loggers=False
    )
except Exception as e:
    logging.error("Cwd must be root of project directory")
logger = logging.Logger(__name__)


def create_map(
    df: pd.DataFrame,
    lat: str,
    lon: str,
    color: str,
    size: Optional[str],
    map_type: str,
    map_style: str,
    zoom: float = 4.5,
    marker_size: int = 5,
    heatmap_size: int = 10,
    color_scale: Optional[str] = None,
    bin_heatmap: bool = False,
) -> go.Figure:
    try:
        logger.info(f"Creating {map_type} map")
        df.loc[:, lat] = pd.to_numeric(df[lat], errors="coerce")
        df.loc[:, lon] = pd.to_numeric(df[lon], errors="coerce")
        df = df.dropna(subset=[lat, lon])
        logger.debug(f"Data shape after cleaning: {df.shape}")

        center_lat = 20.593684
        center_lon = 78.96288

        if map_type == "Scatter":
            fig = px.scatter_mapbox(
                df,
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
                bin_size = 1  # Adjust as needed based on your data distribution
                lat_range = (df["Latitude"].min(), df["Latitude"].max())
                lon_range = (df["Longitude"].min(), df["Longitude"].max())

                # Compute bin edges
                lat_bins = np.arange(lat_range[0], lat_range[1], bin_size)
                lon_bins = np.arange(lon_range[0], lon_range[1], bin_size)

                # Compute mean S4 values within each bin
                mean_s4 = []
                for lat in lat_bins[:-1]:
                    for lon in lon_bins[:-1]:
                        mask = df["Latitude"].between(lat, lat + bin_size) & df[
                            "Longitude"
                        ].between(lon, lon + bin_size)
                        if mask.any():
                            mean_s4.append(df.loc[mask, "S4"].mean())
                        else:
                            mean_s4.append(np.nan)

                # Create a DataFrame for the mean S4 values and corresponding coordinates
                mean_s4_df = pd.DataFrame(
                    {
                        "Latitude": np.repeat(lat_bins[:-1], len(lon_bins[:-1])),
                        "Longitude": np.tile(lon_bins[:-1], len(lat_bins[:-1])),
                        "Mean_S4": mean_s4,
                    }
                )
                # Interpolate missing values
                # Create a grid of coordinates for interpolation
                x, y = np.meshgrid(lon_bins[:-1], lat_bins[:-1])
                z = mean_s4_df["Mean_S4"].values

                # Remove NaN values for interpolation
                valid_idx = ~np.isnan(z)
                x_valid = x.flatten()[valid_idx]
                y_valid = y.flatten()[valid_idx]
                z_valid = z[valid_idx]

                # Perform interpolation using griddata
                interp_method = "cubic"  # Choose interpolation method (e.g., 'linear', 'nearest', 'cubic')
                interp_s4 = griddata(
                    (x_valid, y_valid), z_valid, (x, y), method=interp_method
                )

                # Replace NaN values in original mean_s4_df with interpolated values
                mean_s4_df["Mean_S4"] = interp_s4.flatten()

                fig = px.density_mapbox(
                    mean_s4_df,
                    lat="Latitude",
                    lon="Longitude",
                    z="Mean_S4",
                    radius=heatmap_size,  # Adjust the radius as needed
                    center=dict(
                        lat=df["Latitude"].mean(), lon=df["Longitude"].mean()
                    ),  # Center of the initial view
                    range_color=(0, 1),  # Color range
                    color_continuous_scale=(
                        color_scale if color_scale else None
                    ),  # Color scale
                    title="Square Bin Heatmap of Mean S4 Values",  # Map title
                )
            else:
                fig = px.density_mapbox(
                    df,
                    lat=lat,
                    lon=lon,
                    z=color,
                    radius=heatmap_size,
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
