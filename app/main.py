import logging
import logging.config
import os
import random
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from data_loader import create_file_watcher, load_data
from map_creator import create_map
from visualizations import create_skyplot, create_time_series_plot

try:
    logging.config.fileConfig(
        os.path.join(os.getcwd(), "app", "logging.conf"), disable_existing_loggers=False
    )
except Exception as e:
    logging.error(e)
    logging.error(
        "Cwd must be root of project directory",
    )
logger = logging.Logger(__name__)


def find_time_window(target_datetime, window_minutes=10):
    window_start = target_datetime + timedelta(minutes=0)
    window_end = target_datetime + timedelta(minutes=window_minutes)
    return window_start, window_end


def filter_dataframe(
    df: pd.DataFrame,
    time_window,
    svid,
    latitude_range,
    longitude_range,
    s4_threshold,
):
    if time_window:
        window_start, window_end = time_window
        filtered_df = df[
            (df["IST_Time"] >= window_start)
            & (df["IST_Time"] <= window_end)
            & (df["SVID"].isin(svid))
            & (df["Latitude"] >= latitude_range[0])
            & (df["Latitude"] <= latitude_range[1])
            & (df["Longitude"] >= longitude_range[0])
            & (df["Longitude"] <= longitude_range[1])
            & (df["S4"] >= s4_threshold)
        ]
        return filtered_df
    else:
        filtered_df = df[
            (df["SVID"].isin(svid))
            & (df["Latitude"] >= latitude_range[0])
            & (df["Latitude"] <= latitude_range[1])
            & (df["Longitude"] >= longitude[0])
            & (df["Longitude"] <= longitude[1])
            & (df["S4"] >= s4_threshold)
        ]
    return filtered_df


def find_nearest_time(target_datetime, available_datetimes):
    return min(
        available_datetimes,
        key=lambda t: abs(t - target_datetime),
    )


def main():
    st.set_page_config(layout="wide", page_title="Dynamic S4 Data Plotter")

    st.markdown(
        """
    <style>
        .main {
            padding: 3rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stSelectbox, .stSlider {
            border-radius: 5px;
            padding: 1px;
            margin-bottom: 10px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("Dynamic S4 Data Plotter")

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        data_container = st.empty()
        viz_container = st.empty()
        download_container = st.empty()

        df = load_data(uploaded_file)
        logger.debug(df.columns)
        if df is not None and not df.empty:
            with data_container.container():
                st.write("Data Preview:")
                st.write(df.head())

            st.sidebar.header("Filtering Options")
            svid = st.sidebar.multiselect(
                "Select SVIDs",
                options=sorted(df["SVID"].unique()),
                default=sorted(df["SVID"].unique()),
            )
            unique_datetimes = df["IST_Time"]
            unique_dates = df["IST_Time"].dt.date.unique()
            unique_times = df["IST_Time"].dt.time.unique()
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
                "Time Window", value=10, max_value=30, min_value=5
            )
            selected_datetime = datetime.combine(selected_date, selected_time)
            nearest_datetime = find_nearest_time(selected_datetime, unique_datetimes)
            time_window = find_time_window(nearest_datetime, window)
            latitude_range = st.sidebar.slider("Latitude Range", 0, 90, (0, 90))
            longitude_range = st.sidebar.slider(
                "Longitude Range", -180, 180, (-180, 180)
            )
            s4_threshold = st.sidebar.slider("S4 Threshold", 0.0, 1.0, 0.0)

            filtered_df = filter_dataframe(
                df,
                nearest_datetime,
                svid,
                latitude_range,
                longitude_range,
                s4_threshold,
            )

            # Visualization options
            st.sidebar.header("Visualization Options")
            viz_type = st.sidebar.selectbox(
                "Select Visualization", ["Map", "Time Series", "Skyplot"]
            )

            if viz_type == "Map":
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
                zoom = st.sidebar.slider("Zoom Level", 1.0, 20.0, 4.0)
                marker_size = (
                    st.sidebar.slider("Marker Size", 1, 20, 10)
                    if map_type == "Scatter"
                    else None
                )
                heatmap_size = (
                    st.sidebar.slider("Heatmap Size", 1, 40, 20)
                    if map_type == "Heatmap"
                    else None
                )
                color_scale = st.sidebar.selectbox(
                    "Color Scale", px.colors.named_colorscales(), index=21
                )

                bin_heatmap = (
                    st.sidebar.toggle("Bin heatmap") if map_type == "Heatmap" else None
                )

                fig = create_map(
                    filtered_df,
                    "Latitude",
                    "Longitude",
                    color="S4",
                    size=None,
                    zoom=zoom,
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

            elif viz_type == "Time Series":
                selected_svid = st.sidebar.selectbox(
                    "Select SVID for Time Series", sorted(filtered_df["SVID"].unique())
                )
                filtered_df = filter_dataframe(
                    df,
                    None,
                    svid,
                    latitude_range,
                    longitude_range,
                    s4_threshold,
                )
                time_series_fig = create_time_series_plot(filtered_df, selected_svid)
                with viz_container.container():
                    st.plotly_chart(time_series_fig, use_container_width=True)

            elif viz_type == "Skyplot":
                skyplot_fig = create_skyplot(filtered_df)
                with viz_container.container():
                    st.plotly_chart(skyplot_fig, use_container_width=True)

            with download_container.container():
                if "fig" in locals() and fig.data:
                    unique_key = (
                        f"download_button_{time.time()}_{random.randint(0, 1000000)}"
                    )
                    st.download_button(
                        label="Download Visualization as HTML",
                        data=fig.to_html(),
                        file_name="s4_visualization.html",
                        mime="text/html",
                        key=unique_key,
                    )
                else:
                    st.warning(
                        "Visualization download is not available for this type of plot."
                    )

            def update_viz(new_df):
                with data_container.container():
                    st.write("Data Preview:")
                    st.write(new_df.head())

                filtered_new_df = filter_dataframe(
                    df,
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
                        zoom=zoom,
                        marker_size=marker_size,
                        color_scale=color_scale,
                        bin_heatmap=True,
                    )
                elif viz_type == "Time Series":
                    filtered_new_df = filter_dataframe(
                        df,
                        selected_datetime,
                        svid,
                        latitude_range,
                        longitude_range,
                        s4_threshold,
                    )
                    new_fig = create_time_series_plot(filtered_new_df, selected_svid)
                else:  # Skyplot
                    filtered_df = filter_dataframe(
                        df,
                        selected_datetime,
                        svid,
                        latitude_range,
                        longitude_range,
                        s4_threshold,
                    )
                    new_fig = create_skyplot(filtered_new_df)

                with viz_container.container():
                    if new_fig.data:
                        st.plotly_chart(new_fig, use_container_width=True)
                    else:
                        st.error(
                            "Unable to create visualization. Please check your data and selected options."
                        )

                with download_container.container():
                    if new_fig.data:
                        unique_key = f"download_button_{time.time()}_{random.randint(0, 1000000)}"
                        st.download_button(
                            label="Download Visualization as HTML",
                            data=new_fig.to_html(),
                            file_name="s4_visualization.html",
                            mime="text/html",
                            key=unique_key,
                        )
                    else:
                        st.warning(
                            "Visualization download is not available for this type of plot."
                        )

            file_watcher = create_file_watcher(uploaded_file, update_viz)
            watcher_thread = threading.Thread(target=file_watcher.watch, daemon=True)
            watcher_thread.start()

    st.sidebar.markdown(
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
