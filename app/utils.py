import pandas as pd
from typing import List


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()
