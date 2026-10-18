"""Dataset Profiling Module for EpiWatch Phase 2."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("dataset_profiler")


def profile_dataframe(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Calculate thorough dataset profile metrics for a DataFrame.

    Returns dict with rows, cols, date_range, unique_regions, unique_diseases,
    missing_cells, duplicate_rows, numerical_stats, categorical_stats.
    """
    profile: Dict[str, Any] = {
        "dataset": dataset_name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_range": None,
        "unique_regions": int(df["region"].nunique()) if "region" in df.columns else 0,
        "unique_diseases": int(df["disease"].nunique()) if "disease" in df.columns else 0,
        "missing_by_column": df.isna().sum().to_dict(),
        "numerical_stats": {},
        "categorical_stats": {}
    }

    # Date range discovery
    date_col = "date" if "date" in df.columns else ("month" if "month" in df.columns else None)
    if date_col:
        try:
            d_series = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not d_series.empty:
                profile["date_range"] = {
                    "min_date": str(d_series.min().strftime("%Y-%m-%d")),
                    "max_date": str(d_series.max().strftime("%Y-%m-%d"))
                }
        except Exception:
            pass

    # Numerical statistics
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        series = df[col].dropna()
        if not series.empty:
            profile["numerical_stats"][col] = {
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4)
            }

    # Categorical statistics
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        if col not in ["date", "month"]:
            profile["categorical_stats"][col] = {
                "unique_count": int(df[col].nunique()),
                "top_value": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                "values": df[col].value_counts().head(10).to_dict()
            }

    return profile


def profile_datasets(
    datasets: Dict[str, pd.DataFrame],
    output_dir: Path
) -> Tuple[Path, Path]:
    """Profile all major datasets and save dataset_profile.json and dataset_profile.csv.

    Args:
        datasets: Dict mapping dataset_name -> DataFrame.
        output_dir: Output directory for reports (reports/eda).

    Returns:
        Tuple of (json_path, csv_path).
    """
    logger.info("Profiling all datasets...")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_profiles = {}
    summary_rows = []

    for ds_name, df in datasets.items():
        prof = profile_dataframe(df, ds_name)
        all_profiles[ds_name] = prof

        summary_rows.append({
            "dataset": ds_name,
            "rows": prof["rows"],
            "columns": prof["columns"],
            "missing_cells": prof["total_missing_cells"],
            "duplicates": prof["duplicate_rows"],
            "min_date": prof["date_range"]["min_date"] if prof["date_range"] else None,
            "max_date": prof["date_range"]["max_date"] if prof["date_range"] else None,
            "unique_regions": prof["unique_regions"],
            "unique_diseases": prof["unique_diseases"]
        })

    json_path = output_dir / "dataset_profile.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_profiles, f, indent=2)
    logger.info(f"Saved dataset profile JSON: {json_path}")

    csv_path = output_dir / "dataset_profile.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    logger.info(f"Saved dataset profile CSV: {csv_path}")

    return json_path, csv_path
