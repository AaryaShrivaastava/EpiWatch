"""Reusable CSV Data Loader for EpiWatch platform."""

import os
from pathlib import Path
from typing import Union
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("loader")


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file using pandas, validating existence and extension,

    logging dataset metadata, and returning a pandas DataFrame.

    Args:
        file_path: Path to the CSV file to load.

    Returns:
        pd.DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file is not a valid CSV or is unparseable.
    """
    path = Path(file_path)

    # 1. Existence check
    if not path.exists():
        error_msg = f"Dataset file does not exist: {path.absolute()}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # 2. Extension check
    if path.suffix.lower() != ".csv":
        error_msg = f"Invalid file format: '{path.name}'. File must have a .csv extension."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 3. Reading CSV
    try:
        df = pd.read_csv(path)
    except Exception as e:
        error_msg = f"Failed to parse CSV file '{path.name}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e

    # 4. Logging metadata
    num_rows, num_cols = df.shape
    col_names = list(df.columns)
    logger.info(
        f"Successfully loaded CSV: {path.name} | Rows: {num_rows} | Columns: {num_cols}"
    )
    logger.debug(f"Columns in {path.name}: {col_names}")

    return df
