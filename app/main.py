import streamlit as st
from data_loader import load_data
from map_creator import create_map
from utils import get_numeric_columns, get_categorical_columns


def main():
    st.title("Dynamic Map Plotter")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if not df.empty:
            st.write("Data Preview:")
            st.write(df.head())

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
                ],
            )

            # Additional options
            st.sidebar.header("Map Options")
            zoom = st.sidebar.slider("Zoom Level", 1, 20, 3)
            if map_type in ["Scatter", "Bubble"]:
                marker_size = st.sidebar.slider("Marker Size", 1, 20, 5)
            else:
                marker_size = None

            if color_col in numeric_columns:
                color_scale = st.sidebar.selectbox(
                    "Color Scale", px.colors.named_colorscales()
                )
            else:
                color_scale = None

            fig = create_map(
                df,
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
            st.plotly_chart(fig)

            # Add download button for the map
            st.download_button(
                label="Download Map as HTML",
                data=fig.to_html(),
                file_name="map.html",
                mime="text/html",
            )


if __name__ == "__main__":
    main()
