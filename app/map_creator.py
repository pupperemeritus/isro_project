import plotly.express as px
import pandas as pd
import streamlit as st
from typing import Optional


def create_map(
    df: pd.DataFrame,
    lat: str,
    lon: str,
    color: str,
    size: str,
    map_type: str,
    map_style: str,
    zoom: int = 3,
    marker_size: int = 5,
    color_scale: Optional[str] = None,
) -> Optional[px.Figure]:
    try:
        if map_type == "Scatter":
            fig = px.scatter_mapbox(
                df,
                lat=lat,
                lon=lon,
                color=color,
                size=size,
                zoom=zoom,
                height=600,
                size_max=marker_size,
            )
        elif map_type == "Choropleth":
            fig = px.choropleth_mapbox(
                df, geojson=df, locations=df.index, color=color, zoom=zoom, height=600
            )
        elif map_type == "Bubble":
            fig = px.scatter_geo(
                df,
                lat=lat,
                lon=lon,
                color=color,
                size=size,
                projection="natural earth",
                height=600,
                size_max=marker_size,
            )
        elif map_type == "Heatmap":
            fig = px.density_mapbox(
                df, lat=lat, lon=lon, z=color, radius=10, zoom=zoom, height=600
            )
        else:
            raise ValueError(f"Unsupported map type: {map_type}")

        fig.update_layout(mapbox_style=map_style)

        if color_scale:
            fig.update_traces(marker=dict(colorscale=color_scale))

        return fig
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None
