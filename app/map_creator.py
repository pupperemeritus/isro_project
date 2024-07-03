import plotly.express as px
import pandas as pd
import streamlit as st
from typing import Optional
from plotly.graph_objs import Figure


def create_map(
    df: pd.DataFrame,
    lat: str,
    lon: str,
    color: str,
    size: Optional[str],
    map_type: str,
    map_style: str,
    zoom: int = 3,
    marker_size: int = 5,
    color_scale: Optional[str] = None,
) -> Figure:
    try:
        # Ensure lat and lon are numeric
        df[lat] = pd.to_numeric(df[lat], errors="coerce")
        df[lon] = pd.to_numeric(df[lon], errors="coerce")

        # Drop rows with NaN values in lat or lon
        df = df.dropna(subset=[lat, lon])

        if map_type == "Scatter":
            fig = px.scatter_mapbox(
                df,
                lat=lat,
                lon=lon,
                color=color,
                size=size if size else None,
                zoom=zoom,
                height=600,
                width=800,  # Add width for better visibility
                color_continuous_scale=color_scale if color_scale else None,
            )
            fig.update_traces(marker=dict(size=marker_size))
        elif map_type == "Choropleth":
            fig = px.choropleth_mapbox(
                df,
                geojson=df.to_dict("records"),
                locations=df.index,
                color=color,
                zoom=zoom,
                height=600,
                width=800,
                color_continuous_scale=color_scale if color_scale else None,
            )
        elif map_type == "Bubble":
            fig = px.scatter_geo(
                df,
                lat=lat,
                lon=lon,
                color=color,
                size=size if size else None,
                projection="natural earth",
                height=600,
                width=800,
                size_max=marker_size,
                color_continuous_scale=color_scale if color_scale else None,
            )
        elif map_type == "Heatmap":
            fig = px.density_mapbox(
                df,
                lat=lat,
                lon=lon,
                z=color,
                radius=10,
                zoom=zoom,
                height=600,
                width=800,
                color_continuous_scale=color_scale if color_scale else None,
            )
        else:
            raise ValueError(f"Unsupported map type: {map_type}")

        fig.update_layout(mapbox_style=map_style)
        return fig
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        # Return an empty figure if there's an error
        return Figure()
