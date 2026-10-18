"""EpiWatch Phase 4 — Forecasting Data Loader & Time Series Sanitizer.

Loads Phase 1 master processed/feature datasets and prepares clean, gap-validated
time-series records for every (region, disease) combination.

Key Functions:
--------------
- Sorts chronologically by date.
- Detects duplicate dates, missing dates, and temporal gaps.
- Reindexes series to ensure a strict continuous calendar grid.
- Preserves the distinction between True Zero cases and Missing Observations (NaN).
- Validates minimum series length (e.g. min 28 observations for daily forecasting).
- Logs audit metadata per series.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("forecasting_data_loader")


def load_forecasting_data(
    config_path: str = "configs/forecasting_config.yaml",
    min_observations: int = 28,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Load, clean, and validate time-series data for forecasting.

    Returns
    -------
    clean_df : pd.DataFrame containing sanitized series for all valid (region, disease) pairs
    audit_logs : Dict mapping (region, disease) to summary metadata
    """
    base_config = get_config("config.yaml")

    features_path = base_config.features_dir / "epiwatch_master_features.csv"
    processed_path = base_config.processed_data_dir / "epiwatch_processed_master.csv"

    data_path = features_path if features_path.exists() else processed_path
    if not data_path.exists():
        raise FileNotFoundError(f"Neither {features_path} nor {processed_path} exists. Run Phase 1 first.")

    logger.info(f"Loading forecasting input data from {data_path}...")
    raw_df = pd.read_csv(data_path, parse_dates=["date"], low_memory=False)

    if "region" not in raw_df.columns or "disease" not in raw_df.columns:
        raise ValueError("Dataset must contain 'region' and 'disease' grouping columns.")

    clean_series_list: List[pd.DataFrame] = []
    audit_logs: Dict[str, Dict[str, Any]] = {}

    grouped = raw_df.groupby(["region", "disease"])
    for (region, disease), group in grouped:
        key = f"{region}_{disease}"
        logger.info(f"Processing series: region={region}, disease={disease} ({len(group)} raw rows)...")

        group = group.sort_values("date").reset_index(drop=True)

        # 1. Check duplicate dates
        dup_count = group.duplicated(subset=["date"]).sum()
        if dup_count > 0:
            logger.warning(f"  [{key}] Found {dup_count} duplicate date(s). Keeping last observation.")
            group = group.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        # 2. Determine frequency & grid
        min_date = group["date"].min()
        max_date = group["date"].max()

        full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        expected_count = len(full_date_range)
        actual_count = len(group)
        gaps_count = expected_count - actual_count

        # 3. Reindex to full date grid
        group_indexed = group.set_index("date")
        group_reindexed = group_indexed.reindex(full_date_range)
        group_reindexed.index.name = "date"
        group_reindexed = group_reindexed.reset_index()

        # Fill key metadata columns
        group_reindexed["region"] = region
        group_reindexed["disease"] = disease

        # Calculate missing percentage in target
        target_col = "new_cases"
        missing_count = group_reindexed[target_col].isna().sum()
        missing_pct = round((missing_count / len(group_reindexed)) * 100.0, 2)

        # Safe non-negative constraint
        if target_col in group_reindexed.columns:
            group_reindexed[target_col] = group_reindexed[target_col].clip(lower=0.0)

        # Assess sufficiency
        valid_obs = len(group_reindexed) - missing_count
        insufficient = valid_obs < min_observations

        audit_info = {
            "region": region,
            "disease": disease,
            "number_of_observations": int(valid_obs),
            "expected_grid_length": int(expected_count),
            "gaps_detected": int(gaps_count),
            "missing_percentage": missing_pct,
            "insufficient_history": bool(insufficient),
            "start_date": str(min_date.date()),
            "end_date": str(max_date.date()),
        }
        audit_logs[key] = audit_info

        if insufficient:
            logger.warning(
                f"  [{key}] Insufficient observations ({valid_obs} < {min_observations}). "
                "Marked for Baseline Fallback."
            )
        else:
            logger.info(
                f"  [{key}] Clean series ready — {valid_obs} valid observations, "
                f"{missing_pct}% missing."
            )

        clean_series_list.append(group_reindexed)

    clean_df = pd.concat(clean_series_list, ignore_index=True)
    return clean_df, audit_logs
