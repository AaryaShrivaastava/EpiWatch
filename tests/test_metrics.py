"""Unit tests for forecasting evaluation metrics (MAE, RMSE, Safe MAPE, SMAPE, MASE)."""

import numpy as np
import pytest

from src.forecasting.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_safe_mape,
    calculate_smape,
    calculate_mase,
    evaluate_forecast_metrics,
)


def test_mae_rmse_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)

    assert mae == pytest.approx(2.3333, abs=1e-3)
    assert rmse > mae


def test_safe_mape_zero_handling():
    y_true = np.array([0.0, 0.0, 10.0])
    y_pred = np.array([2.0, 1.0, 12.0])

    mape = calculate_safe_mape(y_true, y_pred)
    assert not np.isnan(mape)
    assert not np.isinf(mape)


def test_smape_range():
    y_true = np.array([10.0, 0.0, 50.0])
    y_pred = np.array([12.0, 0.0, 45.0])

    smape = calculate_smape(y_true, y_pred)
    assert 0.0 <= smape <= 200.0


def test_evaluate_forecast_metrics_dict():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    y_train = np.array([5.0, 10.0, 15.0])

    res = evaluate_forecast_metrics(y_true, y_pred, y_train=y_train)

    assert "MAE" in res
    assert "RMSE" in res
    assert "Safe_MAPE" in res
    assert "SMAPE" in res
    assert "MASE" in res
