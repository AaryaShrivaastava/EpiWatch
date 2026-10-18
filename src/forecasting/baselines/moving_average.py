"""Moving Average Forecaster."""

from typing import Optional
import numpy as np
import pandas as pd


class MovingAverageForecaster:
    """Predicts all future values as the mean of the last W observed values."""

    def __init__(self, window_size: int = 7, name: str = "MovingAverage"):
        self.window_size = window_size
        self.name = f"MovingAverage_{window_size}" if name == "MovingAverage" else name
        self.mean_value_: Optional[float] = None
        self.fitted_ = False

    def fit(self, y: pd.Series) -> "MovingAverageForecaster":
        y_valid = y.dropna().values
        if len(y_valid) == 0:
            self.mean_value_ = 0.0
        else:
            recent = y_valid[-min(len(y_valid), self.window_size):]
            self.mean_value_ = float(np.mean(recent))
        self.fitted_ = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling predict().")
        return np.full(horizon, self.mean_value_, dtype=float)
