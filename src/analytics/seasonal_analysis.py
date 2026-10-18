"""Seasonal Analysis Module for EpiWatch Phase 2.

Note on Seasonality:
--------------------
Statistical seasonality indices capture recurring temporal patterns across calendar months
and weeks. These patterns are analytical associations and do not imply medical or environmental causation.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("seasonal_analysis")


def analyze_seasonality(
    df_epi: pd.DataFrame,
    output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate monthly and weekly seasonality metrics per disease.

    Metrics:
    - month / week_of_year
    - avg_cases, median_cases, std_cases
    - seasonal_index = (period_avg / overall_avg)

    Saves:
    - data/analytics/monthly_seasonality.csv
    - data/analytics/weekly_seasonality.csv
    """
    logger.info("Analyzing monthly and weekly seasonality patterns...")
    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    monthly_rows = []
    weekly_rows = []

    for d, df_d in df.groupby("disease"):
        overall_avg = float(df_d["new_cases"].mean())

        # Monthly aggregation
        m_agg = df_d.groupby("month")["new_cases"].agg(
            avg_cases="mean",
            median_cases="median",
            std_cases="std"
        ).reset_index()

        m_agg["disease"] = d
        m_agg["seasonal_index"] = np.round((m_agg["avg_cases"] / overall_avg) if overall_avg > 0 else 1.0, 4)
        m_agg["avg_cases"] = np.round(m_agg["avg_cases"], 2)
        m_agg["median_cases"] = np.round(m_agg["median_cases"], 2)
        m_agg["std_cases"] = np.round(m_agg["std_cases"].fillna(0.0), 2)

        monthly_rows.append(m_agg)

        # Weekly aggregation
        w_agg = df_d.groupby("week_of_year")["new_cases"].agg(
            avg_cases="mean",
            median_cases="median",
            std_cases="std"
        ).reset_index()

        w_agg["disease"] = d
        w_agg["seasonal_index"] = np.round((w_agg["avg_cases"] / overall_avg) if overall_avg > 0 else 1.0, 4)
        w_agg["avg_cases"] = np.round(w_agg["avg_cases"], 2)
        w_agg["median_cases"] = np.round(w_agg["median_cases"], 2)
        w_agg["std_cases"] = np.round(w_agg["std_cases"].fillna(0.0), 2)

        weekly_rows.append(w_agg)

    m_df = pd.concat(monthly_rows, ignore_index=True)
    w_df = pd.concat(weekly_rows, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    m_csv = output_dir / "monthly_seasonality.csv"
    w_csv = output_dir / "weekly_seasonality.csv"

    m_df.to_csv(m_csv, index=False)
    w_df.to_csv(w_csv, index=False)

    logger.info(f"Saved monthly seasonality CSV: {m_csv}")
    logger.info(f"Saved weekly seasonality CSV: {w_csv}")

    return m_df, w_df
