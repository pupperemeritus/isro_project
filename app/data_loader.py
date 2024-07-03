import pandas as pd
import streamlit as st
from typing import Optional, Callable
import logging
import hashlib
import time
import asyncio

logger = logging.getLogger(__name__)


def load_data(file) -> Optional[pd.DataFrame]:
    try:
        file_content = file.getvalue().decode("utf-8")
        print(f"File content:\n{file_content[:500]}...")  # Print first 500 characters

        file_extension = file.name.split(".")[-1].lower()
        if file_extension == "csv":
            df = pd.read_csv(file)
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        # Basic data cleaning
        df = df.dropna()  # Remove rows with any NaN values
        return df[:100]
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None


class FileWatcher:
    def __init__(self, file, callback: Callable[[pd.DataFrame], None]):
        self.file = file
        self.callback = callback
        self.previous_hash = None
        self.is_running = True

    def watch(self):
        while self.is_running:
            current_hash = hashlib.md5(self.file.getvalue()).hexdigest()
            if current_hash != self.previous_hash:
                self.previous_hash = current_hash
                df = load_data(self.file)
                if df is not None:
                    self.callback(df)
            time.sleep(1)

    def stop(self):
        self.is_running = False


def create_file_watcher(file, callback: Callable[[pd.DataFrame], None]) -> FileWatcher:
    return FileWatcher(file, callback)
