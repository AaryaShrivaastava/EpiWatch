"""SARIMA Forecaster using statsmodels.tsa.statespace.sarimax.SARIMAX."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("sarima_forecaster")


class SARIMAForecaster:
    """Wrapper for statsmodels SARIMA model."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 7),
        name: str = "SARIMA",
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.name = name
        self.model_res_ = None
        self.fitted_ = False

    def fit(self, y: pd.Series) -> "SARIMAForecaster":
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            y_vals = y.ffill().fillna(0.0).values

            model = SARIMAX(
                y_vals,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_res_ = model.fit(disp=False)
            self.fitted_ = True
        except Exception as e:
            logger.warning(f"SARIMA fit failed: {e}. Falling back to Naive fit.")
            self.fitted_ = False
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.fitted_ or self.model_res_ is None:
            raise RuntimeError("SARIMA model is not fitted or failed to fit.")

        forecast_res = self.model_res_.forecast(steps=horizon)
        # Ensure predictions are non-negative
        return np.clip(np.array(forecast_res, dtype=float), 0.0, None)
