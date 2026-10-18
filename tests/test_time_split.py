"""Unit tests for chronological time series split and expanding window rolling validation."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.validation.time_split import chronological_split
from src.forecasting.validation.rolling_validation import expanding_window_generator


def test_chronological_split_ordering():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "new_cases": np.arange(100)})

    train_df, val_df, test_df, meta = chronological_split(df, train_ratio=0.70, val_ratio=0.15)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Strictly chronological
    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()


def test_expanding_window_generator():
    dates = pd.date_range("2023-01-01", periods=150, freq="D")
    df = pd.DataFrame({"date": dates, "new_cases": np.arange(150)})

    folds = list(expanding_window_generator(df, horizon=14, n_folds=3, min_train_size=50))
    assert len(folds) >= 1

    for train_fold, val_fold, fold_num in folds:
        assert len(val_fold) == 14
        assert train_fold["date"].max() < val_fold["date"].min()
