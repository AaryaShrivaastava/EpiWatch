"""Prediction Intervals Calculator (Residual-based uncertainty bounds)."""

from typing import Tuple
import numpy as np


def compute_prediction_intervals(
    y_pred: np.ndarray,
    val_residuals: np.ndarray,
    confidence_level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical prediction intervals around point forecasts.

    Formula:
    --------
    z = 1.96 for 95% confidence level.
    margin_h = z * residual_std * sqrt(1 + h / horizon)

    Returns
    -------
    lower_bound, upper_bound
    """
    y_pred = np.array(y_pred, dtype=float)
    res = np.array(val_residuals, dtype=float)

    res_std = float(np.std(res)) if len(res) > 1 else max(1.0, float(np.mean(np.abs(y_pred)) * 0.15))

    z = 1.96 if confidence_level == 0.95 else 1.645
    horizon = len(y_pred)

    lower_bounds = []
    upper_bounds = []

    for h in range(horizon):
        # Uncertainty expands with horizon step
        scale_factor = np.sqrt(1.0 + (h / max(1, horizon)))
        margin = z * res_std * scale_factor

        low = max(0.0, float(y_pred[h] - margin))
        high = float(y_pred[h] + margin)

        lower_bounds.append(round(low, 2))
        upper_bounds.append(round(high, 2))

    return np.array(lower_bounds), np.array(upper_bounds)
