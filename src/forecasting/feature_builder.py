"""EpiWatch Phase 4 — Time-Series Feature Builder (Leakage-Safe).

Constructs lag, rolling, and temporal features for supervised time-series forecasting.

Strict Leakage Rules:
----------------------
1. All target-derived features (lags, rolling averages, rolling stds) enforce `shift(1)`
   or explicit positive lag offsets. Feature at time T NEVER sees target at T, T+1, etc.
2. Calendar features (day_of_week, month, quarter) are future-known and safe for forecasting.
3. Observed environmental variables (rainfall, temp) are treated as future-unknown at
   horizon > 1 unless lag-shifted.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("forecasting_feature_builder")


def build_forecasting_features(
    df: pd.DataFrame,
    target_col: str = "new_cases",
    lags: List[int] = [1, 2, 3, 7, 14, 21, 28],
    rolling_windows: List[int] = [7, 14, 28],
) -> Tuple[pd.DataFrame, List[str]]:
    """Build leakage-safe supervised forecasting features.

    Returns
    -------
    featured_df : pd.DataFrame with constructed lag/rolling/temporal features
    feature_cols: List[str] of feature column names suitable for training
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "disease", "date"]).reset_index(drop=True)

    feature_cols: List[str] = []

    def _build_group_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        target = group[target_col]

        # 1. Target Lags (strictly lag >= 1)
        for lag in lags:
            col_name = f"target_lag_{lag}"
            group[col_name] = target.shift(lag)

        # 2. Rolling Statistics (strictly shift(1).rolling(window))
        for w in rolling_windows:
            mean_col = f"target_rolling_mean_{w}"
            std_col = f"target_rolling_std_{w}"
            group[mean_col] = target.shift(1).rolling(window=w, min_periods=3).mean()
            group[std_col] = target.shift(1).rolling(window=w, min_periods=3).std().fillna(0.0)

        # 3. Environmental Lags (future-unknown if unlagged, so lag by 1+)
        env_cols = ["temperature_c", "humidity_percent", "rainfall_mm", "air_quality_index"]
        for env in env_cols:
            if env in group.columns:
                group[f"{env}_lag_1"] = group[env].shift(1)
                group[f"{env}_lag_7"] = group[env].shift(7)

        return group

    groups = []
    for _, g in df.groupby(["region", "disease"]):
        groups.append(_build_group_features(g))
    df = pd.concat(groups, axis=0).reset_index(drop=True)

    # 4. Temporal / Calendar Features (Future-Known)
    dates = df["date"]
    df["day_of_week"] = dates.dt.dayofweek
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["month"] = dates.dt.month
    df["quarter"] = dates.dt.quarter
    df["year"] = dates.dt.year

    # Cyclical Sine/Cosine Encodings
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # List all generated feature names
    lag_cols = [f"target_lag_{l}" for l in lags]
    roll_cols = [f"target_rolling_mean_{w}" for w in rolling_windows] + [f"target_rolling_std_{w}" for w in rolling_windows]
    env_lag_cols = [c for c in df.columns if c.endswith("_lag_1") or c.endswith("_lag_7")]
    calendar_cols = ["day_of_week", "week_of_year", "month", "quarter", "year", "sin_month", "cos_month", "sin_dow", "cos_dow"]

    feature_cols = lag_cols + roll_cols + env_lag_cols + calendar_cols

    logger.info(f"Constructed {len(feature_cols)} leakage-safe forecasting features.")
    return df, feature_cols
