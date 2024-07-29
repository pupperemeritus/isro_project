import logging
import logging.config
import random
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import streamlit as st
from data_loader import load_data
from logging_conf import log_config
from map_creator import create_map
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
        pl.col("S4") >= s4_threshold,
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


def main():
    st.set_page_config(layout="wide", page_title="Dynamic S4 Data Plotter")

    st.markdown(
        custom_css_str,
        unsafe_allow_html=True,
    )
    st.title("Dynamic S4 Data Plotter")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a file", type=[".arrow", ".csv", ".parquet"], key="uploaded_file"
    )

    if uploaded_file is not None:
        viz_container = st.empty()
        data_container = st.empty()
        download_container = st.empty()

        df = load_data(uploaded_file)
        if df is not None and not df.is_empty():
            with data_container.container():
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

            # Sidebar inputs for date and time
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

            match viz_type:
                case "Map":
                    map_type = st.sidebar.selectbox(
                        "Select Map Type", ["Scatter", "Heatmap"], index=1
                    )
                    map_style = st.sidebar.selectbox(
                        "Select Map Style",
                        [
                            "open-street-map",
                            "carto-darkmatter",
                            "carto-positron",
                            "stamen- terrain",
                            "stamen-toner",
                            "stamen-watercolor",
                        ],
                    )
                    marker_size = (
                        st.sidebar.slider("Marker Size", 1, 20, 10)
                        if map_type == "Scatter"
                        else None
                    )
                    heatmap_size = (
                        st.sidebar.slider("Heatmap Size", 1, 80, 40)
                        if map_type == "Heatmap"
                        else None
                    )
                    color_scale = st.sidebar.selectbox(
                        "Color Scale", px.colors.named_colorscales(), index=21
                    )

                    bin_heatmap = (
                        st.sidebar.toggle("Bin heatmap")
                        if map_type == "Heatmap"
                        else None
                    )

                    fig = create_map(
                        filtered_df,
                        "Latitude",
                        "Longitude",
                        color="S4",
                        size=None,
                        map_type=map_type,
                        map_style=map_style,
                        marker_size=marker_size,
                        heatmap_size=heatmap_size,
                        color_scale=color_scale,
                        bin_heatmap=bin_heatmap,
                    )

                    with viz_container.container():
                        if fig.data:
                            fig.update_layout(
                                height=1024,  # Adjust this value as needed
                                width=None,  # Allow the width to be responsive
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(
                                "Unable to create map. Please check your data and selected options."
                            )

                case "Time Series":
                    filtered_df = filter_dataframe(
                        df,
                        None,
                        svid,
                        latitude_range,
                        longitude_range,
                        s4_threshold,
                    )
                    time_series_fig = create_time_series_plot(
                        filtered_df, svid
                    )

                    with viz_container.container():
                        st.plotly_chart(time_series_fig, use_container_width=True)

                case "Skyplot":
                    skyplot_fig = create_skyplot(filtered_df)

                    with viz_container.container():
                        st.plotly_chart(skyplot_fig, use_container_width=True)
            with download_container.container():
                match viz_type:
                    case "Map":
                        viz_fig = fig
                    case "Time Series":
                        viz_fig = time_series_fig
                    case "Skyplot":
                        viz_fig = skyplot_fig
                unique_key = (
                    f"download_button_{time.time()}_{random.randint(0, 1000000)}"
                )
                st.download_button(
                    label="Download Visualization as HTML",
                    data=viz_fig.to_html(),
                    file_name="s4_visualization.html",
                    mime="text/html",
                    key=unique_key,
                )

            @st.cache_resource(max_entries=2)
            def update_download_button(fig):
                if fig.data:
                    unique_key = (
                        f"download_button_{time.time()}_{random.randint(0, 1000000)}"
                    )
                    with download_container.container():
                        st.download_button(
                            label="Download Visualization as HTML",
                            data=fig.to_html(),
                            file_name="s4_visualization.html",
                            mime="text/html",
                            key=unique_key,
                        )
                else:
                    with download_container.container():
                        st.warning(
                            "Visualization download is not available for this type of plot."
                        )

            def update_viz(new_df):
                # Only update if the data has changed
                if not new_df.equals(df):
                    with data_container.container():
                        st.write("Data Preview:")
                        st.write(new_df.head())

                    filtered_new_df = filter_dataframe(
                        new_df,
                        selected_datetime,
                        svid,
                        latitude_range,
                        longitude_range,
                        s4_threshold,
                    )

                    if viz_type == "Map":
                        new_fig = create_map(
                            filtered_new_df,
                            "Latitude",
                            "Longitude",
                            "S4",
                            None,
                            map_type,
                            map_style,
                            marker_size=marker_size,
                            color_scale=color_scale,
                            bin_heatmap=True,
                        )
                    elif viz_type == "Time Series":
                        new_fig = create_time_series_plot(
                            filtered_new_df, selected_svid
                        )
                    else:  # Skyplot
                        new_fig = create_skyplot(filtered_new_df)
                    with viz_container.container():
                        if new_fig.data:
                            st.plotly_chart(new_fig, use_container_width=True)
                        else:
                            st.error(
                                "Unable to create visualization. Please check your data and selected options."
                            )

                    # Update download button
                    update_download_button(new_fig)

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
