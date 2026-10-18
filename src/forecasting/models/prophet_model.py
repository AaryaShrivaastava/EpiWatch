"""Prophet Forecaster wrapper (with graceful fallback if uninstalled)."""

from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("prophet_forecaster")


class ProphetForecaster:
    """Wrapper for Facebook Prophet forecaster."""

    def __init__(self, name: str = "Prophet"):
        self.name = name
        self.model_ = None
        self.fitted_ = False
        self.is_available_ = self._check_availability()

    def _check_availability(self) -> bool:
        try:
            import prophet
            return True
        except ImportError:
            logger.info("Prophet unavailable — skipped.")
            return False

    def fit(self, df: pd.DataFrame, date_col: str = "date", target_col: str = "new_cases") -> "ProphetForecaster":
        if not self.is_available_:
            return self

        try:
            from prophet import Prophet
            p_df = pd.DataFrame({
                "ds": pd.to_datetime(df[date_col]),
                "y": df[target_col].fillna(0.0).values
            })
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(p_df)
            self.model_ = model
            self.fitted_ = True
        except Exception as e:
            logger.warning(f"Prophet fit failed: {e}")
            self.fitted_ = False
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self.is_available_ or not self.fitted_ or self.model_ is None:
            raise RuntimeError("Prophet is unavailable or not fitted.")

        future = self.model_.make_future_dataframe(periods=horizon, freq="D")
        forecast = self.model_.predict(future)
        yhat = forecast.iloc[-horizon:]["yhat"].values
        return np.clip(np.array(yhat, dtype=float), 0.0, None)
