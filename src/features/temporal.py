"""Temporal feature engineering for EpiWatch.

Note on Seasonal Categorization:
--------------------------------
The seasonal categorizations (Winter, Summer, Monsoon, Post-monsoon) are standard
climatological feature-engineering buckets for the Indian subcontinent / Bhopal region.
They are constructed for statistical pattern analysis and feature engineering,
and are not claimed as clinical or medical assertions.
"""

from typing import Optional
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("temporal_features")


def get_season(month: int) -> str:
    """Return seasonal label for a given month in India/Bhopal.

    - Winter: December (12), January (1), February (2)
    - Summer: March (3), April (4), May (5)
    - Monsoon: June (6), July (7), August (8), September (9)
    - Post-monsoon: October (10), November (11)
    """
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    elif month in [10, 11]:
        return "Post-monsoon"
    return "Unknown"


def generate_temporal_features(
    df: pd.DataFrame,
    date_col: str = "date"
) -> pd.DataFrame:
    """Generate temporal features: year, month, week, day_of_year, quarter, season.

    Args:
        df: Input DataFrame containing a date/month column.
        date_col: Name of the date column.

    Returns:
        DataFrame with added temporal features.
    """
    logger.info(f"Generating temporal features from column '{date_col}'...")
    df_feat = df.copy()

    actual_date_col = date_col if date_col in df_feat.columns else ("month" if "month" in df_feat.columns else None)

    if not actual_date_col or actual_date_col not in df_feat.columns:
        logger.warning(f"Date column '{date_col}' not found. Skipping temporal feature engineering.")
        return df_feat

    dt_series = pd.to_datetime(df_feat[actual_date_col], errors="coerce")

    df_feat["year"] = dt_series.dt.year
    df_feat["month"] = dt_series.dt.month
    df_feat["week"] = dt_series.dt.isocalendar().week.astype(int)
    df_feat["day_of_year"] = dt_series.dt.dayofyear
    df_feat["quarter"] = dt_series.dt.quarter
    df_feat["season"] = dt_series.dt.month.apply(lambda m: get_season(m) if pd.notna(m) else "Unknown")

    logger.info("Generated temporal features: year, month, week, day_of_year, quarter, season.")
    return df_feat
