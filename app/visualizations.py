import logging
import logging.config
import os

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from logging_conf import log_config

try:
    logging.config.dictConfig(log_config)
except Exception as e:
    logging.error(e)
logger = logging.Logger(__name__)


def create_time_series_plot(df: pl.DataFrame, svid: int) -> go.Figure:
    try:
        logger.info(f"Creating time series plot for SVID {svid}")
        df_svid = df.filter(pl.col("SVID") == svid)
        fig = px.scatter(
            df_svid.to_pandas(),  # Convert to pandas for Plotly
            x="IST_Time",
            height=1024,
            y="S4",
            title=f"S4 Time Series for SVID {svid}",
            range_color=(0, 1),
        )
        logger.debug(f"Time series plot created with {len(df_svid)} points")
        return fig
    except Exception as e:
        logger.exception(f"Error creating time series plot: {str(e)}")
        return go.Figure()


def create_skyplot(df: pl.DataFrame) -> go.Figure:
    try:
        logger.info("Creating skyplot")
        fig = px.scatter_polar(
            df.to_pandas(),  # Convert to pandas for Plotly
            r="Elevation",
            theta="Azimuth",
            color="S4",
            size="S4",
            height=1024,
            range_color=(0, 1),
            hover_data=["SVID", "IST_Time"],
            title="Skyplot with S4 Values",
        )
        fig.update_layout(polar=dict(radialaxis=dict(range=[90, 0])))
        logger.debug(f"Skyplot created with {len(df)} points")
        return fig
    except Exception as e:
        logger.exception(f"Error creating skyplot: {str(e)}")
        return go.Figure()
