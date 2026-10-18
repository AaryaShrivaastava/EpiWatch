"""XGBoost Supervised Time-Series Forecaster (Recursive Multi-Step)."""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("xgboost_forecaster")


class XGBoostForecaster:
    """Supervised time-series forecaster with recursive multi-step predictions."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
        name: str = "XGBoost",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.name = name
        self.model_ = None
        self.feature_cols_: List[str] = []
        self.fitted_ = False

    def _get_regressor(self):
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
            )
        except ImportError:
            logger.warning("XGBoost package not available. Falling back to RandomForestRegressor.")
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, feature_cols: List[str]) -> "XGBoostForecaster":
        self.feature_cols_ = feature_cols
        X_mat = X_train[feature_cols].values.astype(float)
        X_mat = np.nan_to_num(X_mat, nan=0.0)
        y_vec = y_train.values.astype(float)
        y_vec = np.nan_to_num(y_vec, nan=0.0)

        regressor = self._get_regressor()
        regressor.fit(X_mat, y_vec)

        self.model_ = regressor
        self.fitted_ = True
        return self

    def predict_recursive(
        self,
        recent_history_df: pd.DataFrame,
        horizon: int,
        target_col: str = "new_cases",
        lags: List[int] = [1, 2, 3, 7, 14, 21, 28],
        rolling_windows: List[int] = [7, 14, 28],
    ) -> np.ndarray:
        """Perform recursive multi-step forecasting for H steps ahead."""
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("Model is not fitted.")

        history = list(recent_history_df[target_col].dropna().values.astype(float))
        predictions = []

        for h in range(horizon):
            # Construct feature vector for step h from history
            feat_dict = {}

            # Target Lags
            for l in lags:
                feat_dict[f"target_lag_{l}"] = history[-l] if len(history) >= l else history[-1]

            # Rolling Means & Stds
            for w in rolling_windows:
                sub = history[-w:] if len(history) >= w else history
                feat_dict[f"target_rolling_mean_{w}"] = float(np.mean(sub))
                feat_dict[f"target_rolling_std_{w}"] = float(np.std(sub))

            # Environmental Lags
            for env in ["temperature_c", "humidity_percent", "rainfall_mm", "air_quality_index"]:
                for l in [1, 7]:
                    col = f"{env}_lag_{l}"
                    if col in self.feature_cols_:
                        val = recent_history_df[env].dropna().iloc[-l] if (env in recent_history_df.columns and len(recent_history_df) >= l) else 0.0
                        feat_dict[col] = float(val)

            # Calendar features (approximate for future step h)
            last_date = pd.to_datetime(recent_history_df["date"].iloc[-1])
            curr_date = last_date + pd.Timedelta(days=h + 1)

            feat_dict["day_of_week"] = curr_date.dayofweek
            feat_dict["week_of_year"] = curr_date.isocalendar().week
            feat_dict["month"] = curr_date.month
            feat_dict["quarter"] = curr_date.quarter
            feat_dict["year"] = curr_date.year
            feat_dict["sin_month"] = np.sin(2 * np.pi * curr_date.month / 12.0)
            feat_dict["cos_month"] = np.cos(2 * np.pi * curr_date.month / 12.0)
            feat_dict["sin_dow"] = np.sin(2 * np.pi * curr_date.dayofweek / 7.0)
            feat_dict["cos_dow"] = np.cos(2 * np.pi * curr_date.dayofweek / 7.0)

            # Build array in exact feature_cols_ order
            x_row = np.array([[feat_dict.get(c, 0.0) for c in self.feature_cols_]], dtype=float)
            x_row = np.nan_to_num(x_row, nan=0.0)

            pred_val = float(self.model_.predict(x_row)[0])
            pred_val = max(0.0, pred_val)

            predictions.append(pred_val)
            history.append(pred_val)

        return np.array(predictions, dtype=float)
