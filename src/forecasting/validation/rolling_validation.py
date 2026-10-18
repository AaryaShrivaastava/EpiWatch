"""Expanding Window Rolling-Origin Time-Series Cross-Validation."""

from typing import List, Tuple, Generator, Dict, Any
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("rolling_validation")


def expanding_window_generator(
    df: pd.DataFrame,
    date_col: str = "date",
    horizon: int = 28,
    n_folds: int = 3,
    min_train_size: int = 100,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int], None, None]:
    """Generates expanding-window train/validation folds.

    Fold 1: Train = T_0 .. T_n,           Val = T_n+1 .. T_n+h
    Fold 2: Train = T_0 .. T_n + h,       Val = T_n+h+1 .. T_n+2h
    Fold 3: Train = T_0 .. T_n + 2h,     Val = T_n+2h+1 .. T_n+3h

    Yields
    ------
    train_fold_df, val_fold_df, fold_number
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)

    if n < min_train_size + horizon * n_folds:
        # Scale down parameters for small series
        horizon = max(7, (n - min_train_size) // (n_folds + 1))
        min_train_size = min(30, n // 2)

    train_end = n - (horizon * n_folds)
    train_end = max(min_train_size, train_end)

    for fold in range(n_folds):
        curr_train_end = train_end + fold * horizon
        curr_val_end = curr_train_end + horizon

        if curr_val_end > n:
            break

        train_fold = df.iloc[:curr_train_end].reset_index(drop=True)
        val_fold = df.iloc[curr_train_end:curr_val_end].reset_index(drop=True)

        yield train_fold, val_fold, fold + 1
