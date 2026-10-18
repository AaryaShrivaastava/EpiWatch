"""Seasonal Naive Forecaster (y_{t+h} = y_{t+h - S})."""

from typing import Optional
import numpy as np
import pandas as pd


class SeasonalNaiveForecaster:
    """Predicts future values using the value from the corresponding previous seasonal period."""

    def __init__(self, seasonal_period: int = 7, name: str = "SeasonalNaive"):
        self.seasonal_period = seasonal_period
        self.name = name
        self.last_season_: Optional[np.ndarray] = None
        self.fitted_ = False

    def fit(self, y: pd.Series) -> "SeasonalNaiveForecaster":
        y_valid = y.dropna().values
        if len(y_valid) < self.seasonal_period:
            # Fallback to last value if history shorter than season
            self.last_season_ = np.full(self.seasonal_period, y_valid[-1] if len(y_valid) > 0 else 0.0)
        else:
            self.last_season_ = y_valid[-self.seasonal_period:]
        self.fitted_ = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling predict().")

        preds = []
        for h in range(horizon):
            idx = h % self.seasonal_period
            preds.append(self.last_season_[idx])
        return np.array(preds, dtype=float)
