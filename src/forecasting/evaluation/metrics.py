"""Time-Series Forecasting Error Metrics (MAE, RMSE, Safe MAPE, SMAPE, MASE).

Safety Note on Zero-Heavy Epidemiological Data:
----------------------------------------------
MAPE divides by actual values. In epidemiological time series where daily cases can be 0,
standard MAPE returns infinity or division-by-zero errors.
We implement `safe_mape` (which replaces 0 denominator with 1.0 or epsilon) and emphasize
MAE, RMSE, SMAPE, and MASE as robust alternatives.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Safe Mean Absolute Percentage Error for zero-heavy series."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0% to 200%)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 1e-6, np.abs(y_true - y_pred) / denom, 0.0)
    return float(np.mean(ratio) * 100.0)


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonal_period: int = 1) -> float:
    """Mean Absolute Scaled Error relative to in-sample naive/seasonal-naive baseline."""
    mae_pred = np.mean(np.abs(y_true - y_pred))

    if len(y_train) <= seasonal_period:
        denom = 1.0
    else:
        denom = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
        if denom < 1e-6:
            denom = 1.0

    return float(mae_pred / denom)


def evaluate_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonal_period: int = 1,
) -> Dict[str, float]:
    """Calculate all standard forecasting metrics."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    safe_mape = calculate_safe_mape(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)

    if y_train is not None and len(y_train) > 0:
        mase = calculate_mase(y_true, y_pred, y_train, seasonal_period)
    else:
        mase = float(mae)

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Safe_MAPE": round(safe_mape, 4),
        "SMAPE": round(smape, 4),
        "MASE": round(mase, 4),
    }
