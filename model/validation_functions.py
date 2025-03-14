import logging
import polars as pl
from typing import List, Dict, Tuple, Type, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def validate_dataframe_columns(df: pl.DataFrame, required_columns: List[str]) -> None:
    """
    Validates that a Polars DataFrame contains required columns. Raises ValueError if missing.
    """
    if not isinstance(df, pl.DataFrame):
        error_message = "Input is not a Polars DataFrame."
        logger.error(error_message)
        raise TypeError(error_message)

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_message = (
            f"DataFrame is missing required columns: {', '.join(missing_columns)}"
        )
        logger.error(error_message)
        raise ValueError(error_message)


def validate_column_data_types(df: pl.DataFrame, column_types: Dict[str, Type]) -> None:
    """
    Validates data types of DataFrame columns. Raises TypeError if incorrect types.
    """
    if not isinstance(df, pl.DataFrame):
        error_message = "Input is not a Polars DataFrame."
        logger.error(error_message)
        raise TypeError(error_message)

    for column_name, expected_type in column_types.items():
        if column_name not in df.columns:
            logger.warning(
                f"Column '{column_name}' not found, skipping data type validation."
            )
            continue  # Skip validation if column is missing

        actual_dtype = df[column_name].dtype
        if actual_dtype != expected_type:
            error_message = f"Column '{column_name}' has incorrect data type. Expected: {expected_type}, Actual: {actual_dtype}"
            logger.error(error_message)
            raise TypeError(error_message)


def check_nan_and_inf(df: pl.DataFrame, columns: List[str]) -> None:
    """
    Checks for NaN and Inf values in specified DataFrame columns. Logs warnings if found.
    Handles datetime columns separately.
    """
    if not isinstance(df, pl.DataFrame):
        logger.warning("Input is not a Polars DataFrame, skipping NaN/Inf check.")
        return

    for column_name in columns:
        if column_name not in df.columns:
            logger.warning(f"Column '{column_name}' not found, skipping NaN/Inf check.")
            continue

        # Get column dtype
        dtype = df[column_name].dtype

        # Handle datetime columns differently
        if dtype in [pl.Datetime, pl.Date]:
            null_count = df[column_name].is_null().sum()
            if null_count > 0:
                logger.warning(
                    f"Column '{column_name}' contains {null_count} null values."
                )
            continue

        # For numeric columns, check NaN and Inf
        if dtype in [pl.Float32, pl.Float64]:
            nan_count = df[column_name].is_nan().sum()
            inf_count = df[column_name].is_infinite().sum()
            if nan_count > 0 or inf_count > 0:
                logger.warning(
                    f"Column '{column_name}' contains {nan_count} NaN and {inf_count} Inf values."
                )


def check_value_ranges(
    df: pl.DataFrame, column_ranges: Dict[str, Tuple[float, float]]
) -> None:
    """
    Checks if values in DataFrame columns are within given ranges. Logs warnings if outside ranges.
    """
    if not isinstance(df, pl.DataFrame):
        logger.warning("Input is not a Polars DataFrame, skipping value range check.")
        return  # Skip check if not a DataFrame

    for column_name, (min_val, max_val) in column_ranges.items():
        if column_name not in df.columns:
            logger.warning(
                f"Column '{column_name}' not found, skipping value range check."
            )
            continue  # Skip check if column is missing

        min_actual = df[column_name].min()
        max_actual = df[column_name].max()
        if min_actual < min_val or max_actual > max_val:
            logger.warning(
                f"Column '{column_name}' values are outside the expected range [{min_val}, {max_val}]. "
                f"Actual range: [{min_actual}, {max_actual}]"
            )
