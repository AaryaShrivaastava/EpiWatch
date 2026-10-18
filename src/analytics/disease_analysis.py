"""Disease analysis module for EpiWatch Phase 2.

Handles Time-Series Moving Averages, Multi-period Growth Trends, and Year-over-Year (YoY) Analysis.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("disease_analysis")


def classify_trend(growth_rate_percent: float, config_thresholds: Dict[str, float]) -> str:
    """Classify trend category based on growth rate percentage and config thresholds.

    - Rapidly Increasing: > rapid_increase (+30%)
    - Increasing: > increase (+10%)
    - Stable: between stable_low (-10%) and stable_high (+10%)
    - Decreasing: < decrease (-10%)
    - Rapidly Decreasing: < rapid_decrease (-30%)
    """
    if pd.isna(growth_rate_percent):
        return "Unknown"

    rapid_inc = config_thresholds.get("rapid_increase", 30.0)
    inc = config_thresholds.get("increase", 10.0)
    dec = config_thresholds.get("decrease", -10.0)
    rapid_dec = config_thresholds.get("rapid_decrease", -30.0)

    if growth_rate_percent > rapid_inc:
        return "Rapidly Increasing"
    elif growth_rate_percent > inc:
        return "Increasing"
    elif growth_rate_percent < rapid_dec:
        return "Rapidly Decreasing"
    elif growth_rate_percent < dec:
        return "Decreasing"
    else:
        return "Stable"


def analyze_time_series(
    df_epi: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Analyze weekly time-series and calculate 4-week, 8-week, and 12-week moving averages per disease.

    Saves: data/analytics/time_series_moving_averages.csv
    """
    logger.info("Analyzing time-series moving averages per disease...")
    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])

    ts_rows = []
    for d, df_d in df.groupby("disease"):
        df_d = df_d.set_index("date").sort_index()
        weekly = df_d["new_cases"].resample("W-MON").sum().reset_index()
        weekly["disease"] = d

        weekly["cases_4w_ma"] = np.round(weekly["new_cases"].rolling(window=4, min_periods=1).mean(), 2)
        weekly["cases_8w_ma"] = np.round(weekly["new_cases"].rolling(window=8, min_periods=1).mean(), 2)
        weekly["cases_12w_ma"] = np.round(weekly["new_cases"].rolling(window=12, min_periods=1).mean(), 2)

        ts_rows.append(weekly)

    ts_df = pd.concat(ts_rows, ignore_index=True)
    csv_file = output_dir / "time_series_moving_averages.csv"
    ts_df.to_csv(csv_file, index=False)
    logger.info(f"Saved time-series moving averages CSV: {csv_file}")
    return ts_df


def analyze_growth_rates(
    df_epi: pd.DataFrame,
    output_dir: Path,
    config_path: str = "config.yaml"
) -> pd.DataFrame:
    """Calculate 4-week, 8-week, and 12-week growth rates and trend classifications per disease.

    Formula: growth_rate = (current_cases - prev_cases) / max(prev_cases, 1e-5)

    Saves: data/analytics/growth_analysis.csv
    """
    logger.info("Analyzing multi-period growth rates...")
    config = get_config(config_path)
    thresholds = config.get("trend_thresholds", {})

    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])

    growth_rows = []
    for d, df_d in df.groupby("disease"):
        df_d = df_d.set_index("date").sort_index()
        weekly = df_d["new_cases"].resample("W-MON").sum()

        for w_periods in [4, 8, 12]:
            for i in range(w_periods, len(weekly)):
                curr_date = weekly.index[i]
                curr_cases = float(weekly.iloc[i])
                prev_cases = float(weekly.iloc[i - w_periods])

                if prev_cases > 0:
                    g_rate = ((curr_cases - prev_cases) / prev_cases) * 100.0
                else:
                    g_rate = 0.0 if curr_cases == 0 else 100.0

                trend_label = classify_trend(g_rate, thresholds)

                growth_rows.append({
                    "disease": d,
                    "date": curr_date.strftime("%Y-%m-%d"),
                    "period_weeks": w_periods,
                    "current_period_cases": curr_cases,
                    "previous_period_cases": prev_cases,
                    "growth_rate_percent": round(g_rate, 2),
                    "trend_classification": trend_label
                })

    growth_df = pd.DataFrame(growth_rows)
    csv_file = output_dir / "growth_analysis.csv"
    growth_df.to_csv(csv_file, index=False)
    logger.info(f"Saved growth analysis CSV: {csv_file}")
    return growth_df


def analyze_yoy_trends(
    df_epi: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Calculate Year-over-Year (YoY %) case changes per disease.

    Formula: YoY % = (current_year_cases - prev_year_cases) / prev_year_cases * 100

    Saves: data/analytics/yoy_analysis.csv
    """
    logger.info("Analyzing Year-over-Year (YoY) disease trends...")
    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    yoy_rows = []
    yearly_cases = df.groupby(["disease", "year"])["new_cases"].sum().reset_index()

    for d, df_d in yearly_cases.groupby("disease"):
        df_d = df_d.sort_values("year")
        df_d["prev_year_cases"] = df_d["new_cases"].shift(1)

        for _, row in df_d.iterrows():
            curr_y = int(row["year"])
            curr_c = float(row["new_cases"])
            prev_c = row["prev_year_cases"]

            if pd.notna(prev_c) and prev_c > 0:
                yoy_change = ((curr_c - prev_c) / prev_c) * 100.0
            else:
                yoy_change = np.nan

            yoy_rows.append({
                "disease": d,
                "year": curr_y,
                "total_cases": int(curr_c),
                "prev_year_cases": int(prev_c) if pd.notna(prev_c) else None,
                "yoy_change_percent": round(float(yoy_change), 2) if pd.notna(yoy_change) else np.nan
            })

    yoy_df = pd.DataFrame(yoy_rows)
    csv_file = output_dir / "yoy_analysis.csv"
    yoy_df.to_csv(csv_file, index=False)
    logger.info(f"Saved YoY analysis CSV: {csv_file}")
    return yoy_df
