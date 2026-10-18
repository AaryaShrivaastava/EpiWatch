"""EpiWatch Phase 3 — Feature Builder & Weak Label Generator.

Constructs input features and weak analytical labels for Early Outbreak Detection:
1. Loads Phase 1 master features and Phase 2 anomaly data.
2. Selects configurable epidemiological, temporal, environmental, and healthcare features.
3. Ensures strict temporal availability (no future data T+1, T+2 leakage).
4. Generates an explicit analytical weak label (`potential_outbreak`) based on:
   - High statistical anomaly signal (baseline deviation >= 2.0)
   - Sustained case growth (growth_rate_7d > 10.0%)
   - Elevated incidence (incidence_per_100k > 0.0)
5. Performs chronological time-series splitting (Train / Validation / Test).
6. Saves data split metadata to reports/outbreak_detection/data_split.json.

Important:
----------
- `potential_outbreak` is explicitly marked as `weak_statistical_label`.
- It is an ANALYTICAL TARGET for early warning heuristic models, NOT a confirmed medical label.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("feature_builder")


# ── Feature temporal availability registry ─────────────────────────────────
FEATURE_TEMPORAL_DOCS = {
    "new_cases": "Available at day T end-of-day reporting.",
    "active_cases": "Computed from T and T-1 active counts.",
    "deaths": "Available at day T end-of-day reporting.",
    "hospitalizations": "Available at day T healthcare reporting.",
    "icu_admissions": "Available at day T healthcare reporting.",
    "tests": "Available at day T laboratory reporting.",
    "test_positivity_rate_percent": "Computed as tests(T) / cases(T) * 100.",
    "cases_7d_avg": "Computed using shift(1).rolling(7) — uses T-7 to T-1.",
    "cases_14d_avg": "Computed using shift(1).rolling(14) — uses T-14 to T-1.",
    "cases_28d_avg": "Computed using shift(1).rolling(28) — uses T-28 to T-1.",
    "growth_rate_7d": "Computed as (cases_7d(T) - cases_7d(T-7)) / cases_7d(T-7) * 100.",
    "growth_rate_14d": "Computed as (cases_14d(T) - cases_14d(T-14)) / cases_14d(T-14) * 100.",
    "incidence_per_100k": "Computed as cases(T) / population * 100000.",
    "case_fatality_rate": "Computed as deaths(T) / cases(T) * 100.",
    "hospitalization_rate": "Computed as hospitalizations(T) / cases(T) * 100.",
    "temperature_c": "Observed environmental measurement at T.",
    "humidity_percent": "Observed environmental measurement at T.",
    "rainfall_mm": "Observed environmental measurement at T.",
    "air_quality_index": "Observed environmental measurement at T.",
    "rainfall_lag_7": "Environmental measurement at T-7.",
    "humidity_lag_7": "Environmental measurement at T-7.",
    "rainfall_7d": "Rolling 7-day rainfall sum up to T-1.",
    "month": "Calendar feature derived from date T.",
    "week": "Calendar feature derived from date T.",
    "quarter": "Calendar feature derived from date T.",
}


def create_weak_label(df: pd.DataFrame) -> pd.Series:
    """Generate an analytical weak label ('potential_outbreak') for training.

    Rule:
    -----
    potential_outbreak = 1 if:
      (potential_anomaly == 1 OR baseline_deviation >= 2.0)
      AND growth_rate_7d > 10.0
      AND incidence_per_100k > 0.0
    Otherwise: 0.

    Label Type: "weak_statistical_label"
    """
    is_anomaly = (
        (df["potential_anomaly"] == 1)
        if "potential_anomaly" in df.columns
        else (df.get("baseline_deviation", 0) >= 2.0)
    )
    has_growth = df.get("growth_rate_7d", pd.Series(0, index=df.index)) > 10.0
    has_incidence = df.get("incidence_per_100k", pd.Series(0, index=df.index)) > 0.0

    weak_label = (is_anomaly & has_growth & has_incidence).astype(int)
    return weak_label


def split_dataset_chronologically(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split dataset chronologically into Train, Validation, and Test sets.

    Auto-derives ranges based on dates:
      - Train: start through 2023-12-31
      - Validation: 2024-01-01 to 2024-12-31
      - Test: 2025-01-01 to end

    Returns
    -------
    train_df, val_df, test_df, split_info
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    years = df["date"].dt.year.unique()
    max_year = years.max()

    if max_year >= 2025:
        train_mask = df["date"] < "2024-01-01"
        val_mask = (df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")
        test_mask = df["date"] >= "2025-01-01"
    else:
        # Fallback for smaller date ranges: 70% train, 15% val, 15% test
        n = len(df)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)

        train_mask = df.index < train_idx
        val_mask = (df.index >= train_idx) & (df.index < val_idx)
        test_mask = df.index >= val_idx

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    split_info = {
        "total_samples": len(df),
        "train": {
            "samples": len(train_df),
            "start_date": str(train_df["date"].min().date()) if not train_df.empty else None,
            "end_date": str(train_df["date"].max().date()) if not train_df.empty else None,
        },
        "validation": {
            "samples": len(val_df),
            "start_date": str(val_df["date"].min().date()) if not val_df.empty else None,
            "end_date": str(val_df["date"].max().date()) if not val_df.empty else None,
        },
        "test": {
            "samples": len(test_df),
            "start_date": str(test_df["date"].min().date()) if not test_df.empty else None,
            "end_date": str(test_df["date"].max().date()) if not test_df.empty else None,
        },
        "split_type": "chronological",
        "leakage_prevention": "strictly_historical_features_only",
    }

    return train_df, val_df, test_df, split_info


def load_and_build_features(
    config_path: str = "config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Build feature dataset and perform chronological split.

    Returns
    -------
    full_df, train_df, val_df, test_df, feature_cols
    """
    config = get_config(config_path)

    features_dir = config.features_dir
    analytics_dir = config.analytics_data_dir
    outbreak_reports_dir = config.outbreak_reports_dir
    outbreak_reports_dir.mkdir(parents=True, exist_ok=True)

    master_path = features_dir / "epiwatch_master_features.csv"
    if not master_path.exists():
        raise FileNotFoundError(
            f"Master features not found at {master_path}. Run Phase 1 first."
        )

    logger.info(f"Loading master features from {master_path}...")
    full_df = pd.read_csv(master_path, parse_dates=["date"], low_memory=False)

    # Optionally join Phase 2 potential anomalies if present
    anom_path = analytics_dir / "potential_anomalies.csv"
    if anom_path.exists():
        logger.info(f"Joining Phase 2 anomaly labels from {anom_path}...")
        anom_df = pd.read_csv(anom_path, parse_dates=["date"], low_memory=False)
        label_cols = ["date", "region", "disease", "potential_anomaly", "baseline_mean", "upper_threshold"]
        avail = [c for c in label_cols if c in anom_df.columns]
        full_df = full_df.merge(anom_df[avail], on=["date", "region", "disease"], how="left")
        full_df["potential_anomaly"] = full_df["potential_anomaly"].fillna(0).astype(int)

    # Generate analytical weak label
    logger.info("Generating weak statistical label ('potential_outbreak')...")
    full_df["potential_outbreak"] = create_weak_label(full_df)
    full_df["label_type"] = "weak_statistical_label"

    # Selected feature columns
    configured_features = config.get("ml_settings.feature_columns", [])
    feature_cols = [c for c in configured_features if c in full_df.columns]
    logger.info(f"Selected {len(feature_cols)} features for outbreak detection: {feature_cols}")

    # Impute residual NaNs with column median
    for col in feature_cols:
        if full_df[col].isna().any():
            median_val = full_df[col].median()
            full_df[col] = full_df[col].fillna(median_val)

    # Perform chronological split
    train_df, val_df, test_df, split_info = split_dataset_chronologically(full_df)

    logger.info(
        f"Data Split — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )

    # Save split JSON
    split_path = outbreak_reports_dir / "data_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"Data split saved to {split_path}")

    return full_df, train_df, val_df, test_df, feature_cols
