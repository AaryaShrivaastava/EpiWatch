"""Unit tests for prediction interval generation and future forecast record building."""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.prediction.prediction_intervals import compute_prediction_intervals
from src.forecasting.prediction.forecast_generator import generate_future_forecast_records


def test_prediction_intervals_bounds():
    y_pred = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    residuals = np.array([-2.0, 1.5, 0.5, -1.0, 2.0])

    lowers, uppers = compute_prediction_intervals(y_pred, residuals)

    assert len(lowers) == len(y_pred)
    assert len(uppers) == len(y_pred)
    assert (lowers <= y_pred).all()
    assert (uppers >= y_pred).all()
    assert (lowers >= 0.0).all()  # Non-negative case bounds


def test_generate_future_forecast_records():
    y_history = np.array([10.0, 12.0, 14.0, 15.0, 18.0])
    y_pred = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
    val_residuals = np.array([1.0, -1.0, 2.0])
    last_date = pd.Timestamp("2024-12-31")

    rows, summary = generate_future_forecast_records(
        region="Bhopal",
        disease="Dengue",
        last_date=last_date,
        y_history=y_history,
        y_pred=y_pred,
        val_residuals=val_residuals,
        model_name="XGBoost",
        horizon=7,
        baseline_mean=15.0,
    )

    assert len(rows) == 7
    assert rows[0]["date"] == "2025-01-01"
    assert rows[-1]["date"] == "2025-01-07"
    assert summary["trend_direction"] in ["INCREASING", "STABLE", "DECREASING"]
    assert isinstance(summary["forecast_surge_signal"], bool)
