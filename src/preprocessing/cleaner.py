"""Data Cleaning and Standardizing Pipeline for EpiWatch."""

import re
from typing import List, Optional, Tuple
import pandas as pd

from src.preprocessing.missing_values import MissingValueHandler
from src.utils.logging_config import setup_logger

logger = setup_logger("cleaner")


class DataCleaner:
    """Standardizes column names, strips whitespace, parses dates, sorts chronologically,

    removes exact duplicate rows, and handles missing values.
    """

    def __init__(self):
        self.missing_value_handler = MissingValueHandler()

    @staticmethod
    def to_snake_case(name: str) -> str:
        """Convert a string column name to standardized snake_case."""
        name = name.strip()
        # Handle CamelCase -> camel_case
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        # Replace non-alphanumeric chars with underscore
        name = re.sub(r"[^\w\s]", "_", name)
        # Replace whitespace with underscore
        name = re.sub(r"\s+", "_", name)
        # Collapse multiple underscores and strip leading/trailing underscores
        name = re.sub(r"_+", "_", name).strip("_")
        return name.lower()

    def clean_dataset(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        date_col: Optional[str] = "date",
        group_cols: Optional[List[str]] = None,
        data_status: str = "SOURCE_CALIBRATED_SYNTHETIC"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform end-to-end data cleaning on a dataset.

        Args:
            df: Input raw pandas DataFrame.
            dataset_type: Category of dataset ('epidemiological', 'environmental', 'healthcare', 'demographics').
            date_col: Name of date column if applicable.
            group_cols: Columns to group by for chronological sorting and imputation.
            data_status: Data provenance tag ('REAL', 'REAL_HISTORICAL', 'SYNTHETIC', 'SOURCE_CALIBRATED_SYNTHETIC').

        Returns:
            Tuple of (cleaned_df, missing_summary_df)
        """
        logger.info(f"Cleaning dataset of type '{dataset_type}' with shape {df.shape}")
        df_clean = df.copy()

        # 1. Standardize column names to snake_case
        rename_map = {col: self.to_snake_case(col) for col in df_clean.columns}
        df_clean.rename(columns=rename_map, inplace=True)
        logger.debug(f"Standardized column names: {list(df_clean.columns)}")

        # Update date_col if renamed
        actual_date_col = self.to_snake_case(date_col) if date_col else None
        if actual_date_col not in df_clean.columns and "month" in df_clean.columns:
            actual_date_col = "month"

        # 2. Strip whitespace from string/object values
        for col in df_clean.select_dtypes(include=["object", "string"]).columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Restore actual NaNs if string conversion made 'nan' or empty string
            df_clean[col] = df_clean[col].replace(["nan", "NaN", "None", ""], pd.NA)

        # 3. Parse date fields
        if actual_date_col and actual_date_col in df_clean.columns:
            df_clean[actual_date_col] = pd.to_datetime(df_clean[actual_date_col], errors="coerce")
            logger.info(f"Parsed date column '{actual_date_col}' to datetime64.")

        # 4. Remove exact duplicate rows
        duplicates_count = int(df_clean.duplicated().sum())
        if duplicates_count > 0:
            df_clean.drop_duplicates(inplace=True)
            logger.info(f"Removed {duplicates_count} exact duplicate rows.")

        # 5. Sort chronological data
        sort_cols = []
        if actual_date_col and actual_date_col in df_clean.columns:
            sort_cols.append(actual_date_col)
        if group_cols:
            clean_group_cols = [self.to_snake_case(c) for c in group_cols if self.to_snake_case(c) in df_clean.columns]
            # Region or disease first, then date
            for c in clean_group_cols:
                if c not in sort_cols:
                    sort_cols.insert(0, c)

        if sort_cols:
            df_clean.sort_values(by=sort_cols, inplace=True)
            df_clean.reset_index(drop=True, inplace=True)
            logger.info(f"Sorted dataset chronologically by: {sort_cols}")

        # 6. Domain-aware missing value handling
        clean_group_cols = [self.to_snake_case(c) for c in (group_cols or []) if self.to_snake_case(c) in df_clean.columns]
        df_clean, missing_summary_df = self.missing_value_handler.handle_missing_values(
            df_clean, dataset_type=dataset_type, group_cols=clean_group_cols
        )

        # 7. Add Data Provenance Metadata Tag
        if "data_status" not in df_clean.columns:
            df_clean["data_status"] = data_status

        logger.info(f"Cleaned dataset shape: {df_clean.shape}")
        return df_clean, missing_summary_df
