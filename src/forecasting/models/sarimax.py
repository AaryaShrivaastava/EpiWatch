"""SARIMAX Forecaster using statsmodels with Exogenous Features."""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("sarimax_forecaster")


class SARIMAXForecaster:
    """Wrapper for statsmodels SARIMAX model with exogenous variables."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 0, 7),
        exog_cols: Optional[List[str]] = None,
        name: str = "SARIMAX",
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_cols = exog_cols or []
        self.name = name
        self.model_res_ = None
        self.fitted_ = False

    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> "SARIMAXForecaster":
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            y_vals = y.ffill().fillna(0.0).values

            exog_vals = exog[self.exog_cols].fillna(0.0).values if (exog is not None and self.exog_cols) else None

            model = SARIMAX(
                y_vals,
                exog=exog_vals,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_res_ = model.fit(disp=False)
            self.fitted_ = True
        except Exception as e:
            logger.warning(f"SARIMAX fit failed: {e}.")
            self.fitted_ = False
        return self

    def predict(self, horizon: int, exog_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.fitted_ or self.model_res_ is None:
            raise RuntimeError("SARIMAX model is not fitted.")

        exog_vals = exog_future[self.exog_cols].fillna(0.0).values if (exog_future is not None and self.exog_cols) else None
        forecast_res = self.model_res_.forecast(steps=horizon, exog=exog_vals)
        return np.clip(np.array(forecast_res, dtype=float), 0.0, None)
