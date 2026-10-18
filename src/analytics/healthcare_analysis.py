"""Healthcare burden analysis module for EpiWatch Phase 2."""

from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("healthcare_analysis")


def analyze_healthcare_burden(
    master_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Analyze monthly healthcare burden, capacity per 100k, bed/ICU occupancy ratios,

    and correlation with disease case peaks.

    Saves: data/analytics/healthcare_burden_summary.csv
    """
    logger.info("Analyzing healthcare burden metrics...")
    df = master_df.copy()

    # Aggregate by month if daily
    if "date" in df.columns:
        df["month_str"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    elif "month" in df.columns:
        df["month_str"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)

    group_cols = ["month_str"]
    if "region" in df.columns:
        group_cols.append("region")

    agg_dict = {}
    for c in ["new_cases", "hospitalizations", "icu_admissions"]:
        if c in df.columns:
            agg_dict[c] = "sum"

    for c in [
        "hospital_beds", "icu_beds", "doctors", "beds_per_100k",
        "icu_beds_per_100k", "doctors_per_100k", "available_oxygen_percent",
        "medicine_availability_percent", "testing_capacity_tests_per_day"
    ]:
        if c in df.columns:
            agg_dict[c] = "mean"

    burden_df = df.groupby(group_cols).agg(agg_dict).reset_index()

    # Calculate occupancy and utilization
    if "hospitalizations" in burden_df.columns and "hospital_beds" in burden_df.columns:
        # Assuming monthly average daily occupancy or total monthly hospitalizations
        burden_df["estimated_bed_occupancy_ratio"] = np.round(
            np.minimum(burden_df["hospitalizations"] / (burden_df["hospital_beds"] * 30), 1.0), 4
        )

    if "icu_admissions" in burden_df.columns and "icu_beds" in burden_df.columns:
        burden_df["estimated_icu_occupancy_ratio"] = np.round(
            np.minimum(burden_df["icu_admissions"] / (burden_df["icu_beds"] * 30), 1.0), 4
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "healthcare_burden_summary.csv"
    burden_df.to_csv(csv_file, index=False)
    logger.info(f"Saved healthcare burden summary CSV: {csv_file}")
    return burden_df
