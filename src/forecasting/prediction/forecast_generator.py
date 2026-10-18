"""Future Forecast Generator & Surge Signal Evaluator."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.forecasting.prediction.prediction_intervals import compute_prediction_intervals
from src.utils.logging_config import setup_logger

logger = setup_logger("forecast_generator")


def generate_future_forecast_records(
    region: str,
    disease: str,
    last_date: pd.Timestamp,
    y_history: np.ndarray,
    y_pred: np.ndarray,
    val_residuals: np.ndarray,
    model_name: str,
    horizon: int,
    baseline_mean: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generates future forecast records with trend and surge signals.

    Trend Direction:
    ----------------
    growth > +5.0% -> INCREASING
    growth between -5.0% and +5.0% -> STABLE
    growth < -5.0% -> DECREASING

    Forecast Surge Signal:
    ----------------------
    if predicted_cases significantly exceed historical baseline_mean (+20%+):
      forecast_surge_signal = TRUE
      forecast_surge_score = min(100, max(0, (forecast_avg / baseline_mean - 1.0) * 100))

    Returns
    -------
    forecast_rows, summary_indicators
    """
    lower_bounds, upper_bounds = compute_prediction_intervals(y_pred, val_residuals)

    last_observed = float(y_history[-1]) if len(y_history) > 0 else 1.0
    forecast_avg = float(np.mean(y_pred))
    forecast_sum = float(np.sum(y_pred))
    forecast_peak = float(np.max(y_pred))
    peak_step = int(np.argmax(y_pred)) + 1
    peak_date = last_date + pd.Timedelta(days=peak_step)

    # Growth rate relative to recent history
    denom = max(1.0, last_observed)
    forecast_growth_rate = round(((forecast_avg - last_observed) / denom) * 100.0, 2)

    # Trend direction
    if forecast_growth_rate > 5.0:
        trend = "INCREASING"
    elif forecast_growth_rate < -5.0:
        trend = "DECREASING"
    else:
        trend = "STABLE"

    # Future Surge Signal (distinct from current outbreak signal)
    b_mean = max(1.0, baseline_mean)
    surge_diff_pct = ((forecast_avg - b_mean) / b_mean) * 100.0
    surge_signal = bool(surge_diff_pct >= 20.0)
    surge_score = float(np.clip(max(0.0, surge_diff_pct * 2.0), 0.0, 100.0))

    rows = []
    for h in range(horizon):
        curr_date = last_date + pd.Timedelta(days=h + 1)
        rows.append({
            "date": str(curr_date.date()),
            "region": region,
            "disease": disease,
            "horizon": horizon,
            "horizon_step": h + 1,
            "predicted_cases": round(float(y_pred[h]), 2),
            "lower_bound": lower_bounds[h],
            "upper_bound": upper_bounds[h],
            "model": model_name,
            "forecast_growth_rate": forecast_growth_rate,
            "trend_direction": trend,
            "forecast_surge_signal": surge_signal,
            "forecast_surge_score": round(surge_score, 2),
        })

    summary_info = {
        "region": region,
        "disease": disease,
        "horizon": horizon,
        "model": model_name,
        "forecast_total_cases": round(forecast_sum, 2),
        "forecast_peak_cases": round(forecast_peak, 2),
        "forecast_peak_date": str(peak_date.date()),
        "forecast_growth_rate": forecast_growth_rate,
        "trend_direction": trend,
        "forecast_surge_signal": surge_signal,
        "forecast_surge_score": round(surge_score, 2),
    }

    return rows, summary_info
