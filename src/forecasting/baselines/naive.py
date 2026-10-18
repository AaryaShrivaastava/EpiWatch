"""Naive Forecaster (y_{t+h} = y_t)."""

from typing import Optional
import numpy as np
import pandas as pd


class NaiveForecaster:
    """Predicts all future values as the last observed value in the training series."""

    def __init__(self, name: str = "Naive"):
        self.name = name
        self.last_observed_: Optional[float] = None
        self.fitted_ = False

    def fit(self, y: pd.Series) -> "NaiveForecaster":
        y_valid = y.dropna()
        if len(y_valid) == 0:
            self.last_observed_ = 0.0
        else:
            self.last_observed_ = float(y_valid.iloc[-1])
        self.fitted_ = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling predict().")
        return np.full(horizon, self.last_observed_, dtype=float)
