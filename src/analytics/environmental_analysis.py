"""Environmental correlation and matrix analysis module for EpiWatch Phase 2."""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("environmental_analysis")


def analyze_environmental_correlations(
    master_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations between disease cases and lagged environmental variables.

    Lagged variables evaluated:
    - rainfall: 7, 14, 21, 28 days
    - humidity: 7, 14 days
    - temperature: 7, 14 days

    Saves: data/analytics/environmental_correlations.csv
    """
    logger.info("Calculating Pearson and Spearman environmental lag correlations...")
    df = master_df.copy()

    # Lag configurations to check
    lag_config = {
        "rainfall_mm": [7, 14, 21, 28],
        "humidity_percent": [7, 14],
        "temperature_c": [7, 14]
    }

    corr_rows = []
    diseases = df["disease"].unique() if "disease" in df.columns else ["All"]

    for d in diseases:
        df_d = df[df["disease"] == d].copy() if "disease" in df.columns else df.copy()

        # Ensure date sorted
        if "date" in df_d.columns:
            df_d["date"] = pd.to_datetime(df_d["date"])
            df_d.sort_values("date", inplace=True)

        for env_feature, lags in lag_config.items():
            if env_feature not in df_d.columns:
                continue

            for lag in lags:
                lag_col = f"{env_feature}_lag_{lag}"
                # Create lag dynamically if not pre-computed
                if lag_col not in df_d.columns:
                    df_d[lag_col] = df_d[env_feature].shift(lag)

                valid_data = df_d[["new_cases", lag_col]].dropna()
                if len(valid_data) > 5:
                    p_corr = valid_data["new_cases"].corr(valid_data[lag_col], method="pearson")
                    s_corr = valid_data["new_cases"].corr(valid_data[lag_col], method="spearman")

                    corr_rows.append({
                        "disease": d,
                        "feature": env_feature,
                        "lag_days": lag,
                        "pearson_correlation": round(float(p_corr), 4) if pd.notna(p_corr) else np.nan,
                        "spearman_correlation": round(float(s_corr), 4) if pd.notna(s_corr) else np.nan
                    })

    corr_df = pd.DataFrame(corr_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "environmental_correlations.csv"
    corr_df.to_csv(csv_file, index=False)
    logger.info(f"Saved environmental correlations CSV: {csv_file}")
    return corr_df


def calculate_correlation_matrices(
    master_df: pd.DataFrame,
    output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate correlation matrices for:

    A. Epidemiological variables
    B. Environmental variables
    C. Combined variables

    Saves matrix CSVs to data/analytics/
    """
    logger.info("Computing correlation matrices...")
    df = master_df.copy()

    epi_cols = [c for c in [
        "new_cases", "active_cases", "recovered", "deaths",
        "hospitalizations", "icu_admissions", "tests", "cases_7d_avg", "incidence_per_100k"
    ] if c in df.columns]

    env_cols = [c for c in [
        "temperature_c", "rainfall_mm", "humidity_percent", "air_quality_index",
        "wind_speed_kmh", "rainfall_7d", "humidity_7d_avg", "temperature_7d_avg"
    ] if c in df.columns]

    combined_cols = list(set(epi_cols + env_cols))

    epi_corr = df[epi_cols].corr(method="pearson").round(4) if epi_cols else pd.DataFrame()
    env_corr = df[env_cols].corr(method="pearson").round(4) if env_cols else pd.DataFrame()
    comb_corr = df[combined_cols].corr(method="pearson").round(4) if combined_cols else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    epi_corr.to_csv(output_dir / "epidemiological_correlation_matrix.csv")
    env_corr.to_csv(output_dir / "environmental_correlation_matrix.csv")
    comb_corr.to_csv(output_dir / "combined_correlation_matrix.csv")

    logger.info("Saved correlation matrices CSVs.")
    return epi_corr, env_corr, comb_corr
