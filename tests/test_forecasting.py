"""End-to-end integration unit tests for Phase 4 forecasting models and pipelines."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines.naive import NaiveForecaster
from src.forecasting.baselines.seasonal_naive import SeasonalNaiveForecaster
from src.forecasting.baselines.moving_average import MovingAverageForecaster
from src.forecasting.models.sarima import SARIMAForecaster
from src.forecasting.models.xgboost_forecaster import XGBoostForecaster
from src.forecasting.feature_builder import build_forecasting_features


def test_baseline_models():
    y = pd.Series([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])

    naive = NaiveForecaster().fit(y)
    p_naive = naive.predict(5)
    assert len(p_naive) == 5
    assert (p_naive == 40.0).all()

    snaive = SeasonalNaiveForecaster(seasonal_period=3).fit(y)
    p_snaive = snaive.predict(5)
    assert len(p_snaive) == 5

    ma = MovingAverageForecaster(window_size=3).fit(y)
    p_ma = ma.predict(5)
    assert len(p_ma) == 5
    assert p_ma[0] == pytest.approx(35.0)


def test_feature_builder_leakage_safety():
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "region": "Bhopal",
        "disease": "Dengue",
        "new_cases": np.arange(50) + 5,
        "temperature_c": np.random.uniform(20, 30, 50),
        "humidity_percent": np.random.uniform(50, 80, 50),
        "rainfall_mm": np.random.uniform(0, 10, 50),
        "air_quality_index": np.random.uniform(100, 200, 50),
    })

    feat_df, cols = build_forecasting_features(df, lags=[1, 7], rolling_windows=[7])

    assert "target_lag_1" in cols
    assert "target_rolling_mean_7" in cols

    # Verify shift(1) leakage safety: lag_1 at index 1 is value from index 0
    assert feat_df["target_lag_1"].iloc[1] == df["new_cases"].iloc[0]


def test_xgboost_recursive_forecasting():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "region": "Bhopal",
        "disease": "Dengue",
        "new_cases": np.random.randint(5, 50, 60),
        "temperature_c": np.random.uniform(20, 30, 60),
        "humidity_percent": np.random.uniform(50, 80, 60),
        "rainfall_mm": np.random.uniform(0, 10, 60),
        "air_quality_index": np.random.uniform(100, 200, 60),
    })

    feat_df, cols = build_forecasting_features(df, lags=[1, 7], rolling_windows=[7])

    xgb_model = XGBoostForecaster(n_estimators=10, max_depth=3)
    xgb_model.fit(feat_df.iloc[:40], feat_df["new_cases"].iloc[:40], cols)

    preds = xgb_model.predict_recursive(feat_df.iloc[:40], horizon=7, lags=[1, 7], rolling_windows=[7])

    assert len(preds) == 7
    assert (preds >= 0.0).all()
