import streamlit as st
import asyncio
from data_loader import load_data, create_file_watcher
from map_creator import create_map
from utils import get_numeric_columns, get_categorical_columns
import plotly.express as px
import time
import random


async def main():
    st.set_page_config(layout="wide", page_title="Dynamic Map Plotter")

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
    st.title("Dynamic Map Plotter")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        data_container = st.empty()
        map_container = st.empty()
        download_container = st.empty()

        df = load_data(uploaded_file)

        if df is not None and not df.empty:
            # Initial data preview
            with data_container.container():
                st.write("Data Preview:")
                st.write(df.head())

            # Static UI elements
            numeric_columns = get_numeric_columns(df)
            categorical_columns = get_categorical_columns(df)

            lat_col = st.selectbox("Select Latitude Column", numeric_columns)
            lon_col = st.selectbox("Select Longitude Column", numeric_columns)
            color_col = st.selectbox("Select Color Column", df.columns)
            size_col = st.selectbox("Select Size Column", numeric_columns, index=None)

            map_type = st.selectbox(
                "Select Map Type", ["Scatter", "Choropleth", "Bubble", "Heatmap"]
            )
            map_style = st.selectbox(
                "Select Map Style",
                [
                    "open-street-map",
                    "carto-positron",
                    "carto-darkmatter",
                    "stamen-terrain",
                    "stamen-toner",
                    "stamen-watercolor",
                    "white-bg",
                ],
            )

            st.sidebar.header("Map Options")
            zoom = st.sidebar.slider("Zoom Level", 1, 20, 3)
            marker_size = (
                st.sidebar.slider("Marker Size", 1, 20, 5)
                if map_type in ["Scatter", "Bubble"]
                else None
            )
            color_scale = (
                st.sidebar.selectbox("Color Scale", px.colors.named_colorscales())
                if color_col in numeric_columns
                else None
            )

            def update_map(new_df):
                with data_container.container():
                    st.write("Data Preview:")
                    st.write(new_df.head())

                fig = create_map(
                    new_df,
                    lat_col,
                    lon_col,
                    color_col,
                    size_col,
                    map_type,
                    map_style,
                    zoom=zoom,
                    marker_size=marker_size,
                    color_scale=color_scale,
                )

                with map_container.container():
                    if fig.data:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(
                            "Unable to create map. Please check your data and selected options."
                        )

                with download_container.container():
                    if fig.data:
                        # Generate a unique key using timestamp and random number
                        unique_key = f"download_button_{time.time()}_{random.randint(0, 1000000)}"
                        st.download_button(
                            label="Download Map as HTML",
                            data=fig.to_html(),
                            file_name="map.html",
                            mime="text/html",
                            key=unique_key,
                        )
                    else:
                        st.warning(
                            "Map download is not available due to an error in map creation."
                        )

            # Initial map creation
            update_map(df)

            # Create and start file watcher
            file_watcher = create_file_watcher(uploaded_file, update_map)
            await file_watcher.watch()


if __name__ == "__main__":
    asyncio.run(main())
