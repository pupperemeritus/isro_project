import io
import logging
import logging.config
import random
import os
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import streamlit as st
from custom_css import custom_css_str
from data_loader import load_data
from logging_conf import log_config
from map_creator import create_contour_map, create_map
from visualizations import create_skyplot, create_time_series_plot

try:
    logging.config.dictConfig(log_config)
except Exception as e:
    logging.error(e)
logger = logging.Logger(__name__)


def find_time_window(target_datetime, window_minutes=10):
    window_start = target_datetime + timedelta(minutes=0)
    window_end = target_datetime + timedelta(minutes=window_minutes)
    return window_start, window_end


def filter_dataframe(
    df: pl.DataFrame,
    time_window,
    svid,
    latitude_range,
    longitude_range,
    s4_threshold,
):
    filter_conditions = [
        pl.col("SVID").is_in(svid),
        pl.col("Latitude").is_between(latitude_range[0], latitude_range[1]),
        pl.col("Longitude").is_between(longitude_range[0], longitude_range[1]),
        pl.col("Vertical S4") >= s4_threshold,
    ]

    if time_window:
        window_start, window_end = time_window
        filter_conditions.extend(
            [pl.col("IST_Time") >= window_start, pl.col("IST_Time") <= window_end]
        )

    # Apply all filter conditions
    for condition in filter_conditions:
        df = df.filter(condition)

    return df


def find_nearest_time(
    target_datetime: datetime, available_datetimes: pl.Series
) -> datetime:
    time_diffs = (available_datetimes - target_datetime).abs()
    nearest_index = time_diffs.arg_min()
    return available_datetimes[nearest_index]


def save_matplotlib_figure_as_png(fig: plt.Figure):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return buffer


# Function to save Plotly figure as PNG
def save_plotly_figure_as_png(fig: go.Figure):
    buffer = io.BytesIO()
    pio.write_image(fig, buffer, format="png")
    buffer.seek(0)
    return buffer


def create_fig(df, viz_type, **kwargs):
    if viz_type == "Map":
        map_type = kwargs.get("map_type", "Scatter/Heatmap")
        map_fig = None
        if map_type == "TEC":
            map_fig = create_contour_map(
                df,
                lat="Latitude",
                lon="Longitude",
                color="Vertical S4",
            )
        elif map_type == "Scatter/Heatmap":
            map_fig = create_map(
                df,
                "Latitude",
                "Longitude",
                color="Vertical S4",
                size=40,
                **kwargs,
            )
        return map_fig

    elif viz_type == "Time Series":
        return create_time_series_plot(df, kwargs.get("svid"))
    elif viz_type == "Skyplot":
        return create_skyplot(df)
    else:
        raise ValueError(f"Unsupported visualization type: {viz_type}")


def main():
    st.set_page_config(layout="wide", page_title="Dynamic S4 Data Plotter")
    st.markdown(custom_css_str, unsafe_allow_html=True)

    data_dir = os.path.join(os.getcwd(), "data/")
    available_files = os.listdir(data_dir)
    available_files_path = [
        os.path.join(data_dir, avl_file) for avl_file in available_files
    ]

    current_file_selection = st.sidebar.selectbox(
        "Select file", options=available_files
    )
    current_index = available_files.index(current_file_selection)
    current_file = io.FileIO(available_files_path[current_index])

    if current_file is not None:
        df = load_data(current_file)
        if df is not None and not df.is_empty():
            viz_container = st.empty()
            data_container = st.empty()
            download_container = st.empty()
            with data_container.container(height=300, border=False):
                st.write("Data Preview:")
                st.write(df.head(3))

            st.sidebar.header("Filtering Options and Visualization Options")

            unique_dates = (
                df.select(pl.col("IST_Time").dt.date())
                .unique()
                .to_series()
                .sort()
                .to_list()
            )
            unique_times = (
                df.select(pl.col("IST_Time").dt.time())
                .unique()
                .to_series()
                .sort()
                .to_list()
            )

            selected_date = st.sidebar.date_input(
                "Select Date",
                value=unique_dates[0],
                min_value=min(unique_dates),
                max_value=max(unique_dates),
            )

            selected_time = st.sidebar.slider(
                "Select Time",
                value=unique_times[0],
                min_value=min(unique_times),
                max_value=max(unique_times),
                step=timedelta(minutes=1),
            )
            window = st.sidebar.slider(
                "Time Window", value=10, max_value=30, min_value=1
            )

            unique_datetimes = (
                df.select(pl.col("IST_Time").dt.replace_time_zone(None))
                .unique()
                .to_series()
                .sort()
            )
            selected_datetime = datetime.combine(selected_date, selected_time)
            nearest_datetime = find_nearest_time(selected_datetime, unique_datetimes)
            time_window = find_time_window(nearest_datetime, window)

            latitude_range = st.sidebar.slider("Latitude Range", 0, 90, (0, 90))
            longitude_range = st.sidebar.slider(
                "Longitude Range", -180, 180, (-180, 180)
            )
            s4_threshold = st.sidebar.slider("S4 Threshold", 0.0, 1.0, 0.0)
            # Visualization options
            viz_type = st.sidebar.selectbox(
                "Select Visualization", ["Map", "Time Series", "Skyplot"]
            )
            svid = st.sidebar.multiselect(
                "Select SVIDs",
                options=sorted(df["SVID"].unique().to_list()),
                default=sorted(df["SVID"].unique().to_list()),
            )

            filtered_df = filter_dataframe(
                df,
                time_window,
                svid,
                latitude_range,
                longitude_range,
                s4_threshold,
            )

            viz_options = {}
            if viz_type == "Map":
                viz_options["map_type"] = st.sidebar.selectbox(
                    "Select Map Type", ["Scatter/Heatmap", "TEC"], index=0
                )
                viz_options["map_style"] = st.sidebar.selectbox(
                    "Select Map Style",
                    [
                        "open-street-map",
                        "carto-darkmatter",
                        "carto-positron",
                        "stamen-terrain",
                        "stamen-toner",
                        "stamen-watercolor",
                    ],
                )
                if viz_options["map_type"] == "Scatter/Heatmap":
                    viz_options["marker_size"] = st.sidebar.slider(
                        "Marker Size", 1, 20, 10
                    )
                    viz_options["heatmap_size"] = st.sidebar.slider(
                        "Heatmap Size", 1, 80, 40
                    )
                    viz_options["color_scale"] = st.sidebar.selectbox(
                        "Color Scale", px.colors.named_colorscales(), index=21
                    )
                    viz_options["bin_heatmap"] = st.sidebar.toggle("Bin heatmap")
            elif viz_type == "Time Series":
                viz_options["svid"] = st.selectbox(
                    "Select SVID", options=sorted(df["SVID"].unique().to_list())
                )

            fig = create_fig(filtered_df, viz_type, **viz_options)

            # Use the appropriate Streamlit function based on the figure type
            with viz_container.container(border=False):
                if isinstance(fig, plt.Figure):
                    st.pyplot(fig, use_container_width=True)
                elif isinstance(fig, go.Figure):
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Unsupported figure type")

            buffer = io.BytesIO()

            with download_container.container(height=44, border=False):
                st.download_button(
                    label="Download Visualization as PNG",
                    data=buffer,
                    file_name=f"s4_visualization_{viz_type}_{viz_options.get('map_type', '')}.png",
                    mime="image/png",
                )
    st.markdown(
        """
    ## Help
    - **Map**: Shows S4 data on a map. Use 'Scatter' for individual points or 'Heatmap' for density.
    - **Time Series**: Displays S4 values over time for a selected satellite.
    - **Skyplot**: Shows satellite positions with color-coded S4 values.
    - Use the filters to narrow down the data shown in the visualizations.
    """
    )


if __name__ == "__main__":
    main()
