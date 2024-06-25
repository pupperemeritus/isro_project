import pandas as pd
import streamlit as st
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_data(file) -> Optional[pd.DataFrame]:
    try:
        file_extension = file.name.split(".")[-1].lower()
        if file_extension == "csv":
            df = pd.read_csv(file)
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Basic data cleaning
        df = df.dropna()  # Remove rows with any NaN values
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None
