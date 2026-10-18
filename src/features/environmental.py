"""Environmental feature engineering for EpiWatch.

Data Leakage Prevention:
------------------------
Environmental lags and rolling averages are calculated strictly using past observations.
Grouped by region, features use shift(N) for lags and rolling(window, min_periods=1) for
rolling aggregates to ensure no future weather/environmental information leaks into the dataset.
"""

from typing import List, Optional
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("environmental_features")


def generate_environmental_features(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Generate lagged and rolling environmental features.

    Features generated (if columns exist):
    - Lags: rainfall_lag_7, rainfall_lag_14, rainfall_lag_21,
            humidity_lag_7, humidity_lag_14, temperature_lag_7
    - Rolling: rainfall_7d, rainfall_14d, humidity_7d_avg, temperature_7d_avg

    Args:
        df: Cleaned environmental DataFrame.
        group_cols: Grouping columns (e.g. ['region']).

    Returns:
        DataFrame with added environmental lag and rolling features.
    """
    logger.info("Generating environmental features...")
    df_feat = df.copy()

    # Ensure chronological order
    if "date" in df_feat.columns:
        df_feat["date"] = pd.to_datetime(df_feat["date"])
        sort_keys = [c for c in (group_cols or ["region"]) if c in df_feat.columns] + ["date"]
        df_feat.sort_values(by=sort_keys, inplace=True)
        df_feat.reset_index(drop=True, inplace=True)

    g_cols = [c for c in (group_cols or ["region"]) if c in df_feat.columns]

    # Map column names if needed
    rain_col = "rainfall_mm" if "rainfall_mm" in df_feat.columns else ("rainfall" if "rainfall" in df_feat.columns else None)
    hum_col = "humidity_percent" if "humidity_percent" in df_feat.columns else ("humidity" if "humidity" in df_feat.columns else None)
    temp_col = "temperature_c" if "temperature_c" in df_feat.columns else ("temperature" if "temperature" in df_feat.columns else None)

    # 1. Rainfall Lag Features
    if rain_col:
        for lag in [7, 14, 21]:
            col_name = f"rainfall_lag_{lag}"
            if g_cols:
                df_feat[col_name] = df_feat.groupby(g_cols)[rain_col].shift(lag)
            else:
                df_feat[col_name] = df_feat[rain_col].shift(lag)
            logger.info(f"Generated feature: {col_name}")

        # Rainfall Rolling Sums
        for window in [7, 14]:
            col_name = f"rainfall_{window}d"
            if g_cols:
                df_feat[col_name] = df_feat.groupby(g_cols)[rain_col].transform(
                    lambda s: s.rolling(window=window, min_periods=1).sum()
                )
            else:
                df_feat[col_name] = df_feat[rain_col].rolling(window=window, min_periods=1).sum()
            logger.info(f"Generated feature: {col_name}")

    # 2. Humidity Lag & Rolling Features
    if hum_col:
        for lag in [7, 14]:
            col_name = f"humidity_lag_{lag}"
            if g_cols:
                df_feat[col_name] = df_feat.groupby(g_cols)[hum_col].shift(lag)
            else:
                df_feat[col_name] = df_feat[hum_col].shift(lag)
            logger.info(f"Generated feature: {col_name}")

        col_name = "humidity_7d_avg"
        if g_cols:
            df_feat[col_name] = df_feat.groupby(g_cols)[hum_col].transform(
                lambda s: s.rolling(window=7, min_periods=1).mean()
            )
        else:
            df_feat[col_name] = df_feat[hum_col].rolling(window=7, min_periods=1).mean()
        logger.info(f"Generated feature: {col_name}")

    # 3. Temperature Lag & Rolling Features
    if temp_col:
        col_name = "temperature_lag_7"
        if g_cols:
            df_feat[col_name] = df_feat.groupby(g_cols)[temp_col].shift(7)
        else:
            df_feat[col_name] = df_feat[temp_col].shift(7)
        logger.info(f"Generated feature: {col_name}")

        col_name = "temperature_7d_avg"
        if g_cols:
            df_feat[col_name] = df_feat.groupby(g_cols)[temp_col].transform(
                lambda s: s.rolling(window=7, min_periods=1).mean()
            )
        else:
            df_feat[col_name] = df_feat[temp_col].rolling(window=7, min_periods=1).mean()
        logger.info(f"Generated feature: {col_name}")

    return df_feat
