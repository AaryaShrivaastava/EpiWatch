"""Chronological Train / Validation / Test Time-Series Splitter."""

from typing import Tuple, Dict, Any
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("time_split")


def chronological_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Splits a single time series or DataFrame chronologically.

    Rule: NEVER use random splitting for time series.

    Returns
    -------
    train_df, val_df, test_df, metadata
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)

    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end_idx].reset_index(drop=True)
    val_df = df.iloc[train_end_idx:val_end_idx].reset_index(drop=True)
    test_df = df.iloc[val_end_idx:].reset_index(drop=True)

    split_meta = {
        "total_observations": n,
        "train": {
            "samples": len(train_df),
            "start": str(train_df[date_col].min().date()) if not train_df.empty else None,
            "end": str(train_df[date_col].max().date()) if not train_df.empty else None,
        },
        "validation": {
            "samples": len(val_df),
            "start": str(val_df[date_col].min().date()) if not val_df.empty else None,
            "end": str(val_df[date_col].max().date()) if not val_df.empty else None,
        },
        "test": {
            "samples": len(test_df),
            "start": str(test_df[date_col].min().date()) if not test_df.empty else None,
            "end": str(test_df[date_col].max().date()) if not test_df.empty else None,
        },
        "split_method": "strict_chronological_ratio",
    }

    return train_df, val_df, test_df, split_meta
