import logging
import logging.config
import os

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


def create_time_series_plot(df: pd.DataFrame, svid: int) -> go.Figure:
    try:
        logger.info(f"Creating time series plot for SVID {svid}")
        df_svid = df[df["SVID"] == svid]
        fig = px.scatter(
            df_svid,
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


def create_skyplot(df: pd.DataFrame) -> go.Figure:
    try:
        logger.info("Creating skyplot")
        fig = px.scatter_polar(
            df,
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
