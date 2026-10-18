"""EpiWatch Phase 3 — Level 1: Historical Baseline Detector.

Computes a statistical historical baseline for each (region, disease) series:
  - baseline_mean: historical rolling mean (28-day window by default)
  - baseline_std: historical rolling standard deviation
  - upper_threshold: baseline_mean + 2 * baseline_std
  - critical_threshold: baseline_mean + 3 * baseline_std
  - baseline_deviation: (new_cases - baseline_mean) / baseline_std

Safety:
-------
Zero standard deviation is handled safely by setting baseline_deviation to 0.0
to avoid division by zero or silent overflow.

Outputs:
--------
- baseline_anomaly_score: 0–100 continuous score
- statistical_anomaly: "normal", "potential_anomaly", or "strong_anomaly"

Disclaimer:
-----------
These thresholds are statistical heuristics, NOT medically confirmed standard boundaries.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("baseline_detector")


def detect_baseline_anomalies(
    df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """Compute Level 1 historical baseline thresholds and anomaly categories.

    Parameters
    ----------
    df          : DataFrame containing date, region, disease, and new_cases
    config_path : Path to config.yaml

    Returns
    -------
    out_df : DataFrame with added Level 1 baseline anomaly columns
    """
    config = get_config(config_path)

    window = int(config.get("outbreak_detection.baseline.rolling_window_days", 28))
    upper_mult = float(config.get("outbreak_detection.baseline.upper_std_multiplier", 2.0))
    crit_mult = float(config.get("outbreak_detection.baseline.critical_std_multiplier", 3.0))

    logger.info(f"Computing Level 1 Historical Baseline (window={window}d, upper={upper_mult}std, critical={crit_mult}std)...")


    out_df = df.copy()
    out_df = out_df.sort_values(["region", "disease", "date"]).reset_index(drop=True)

    def _calc_group_baseline(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        cases = group["new_cases"]

        # Shift(1) ensures no future data is included in historical mean/std
        b_mean = cases.shift(1).rolling(window=window, min_periods=7).mean()
        b_std = cases.shift(1).rolling(window=window, min_periods=7).std()

        # Fallback for initial window
        b_mean = b_mean.fillna(cases.expanding().mean())
        b_std = b_std.fillna(cases.expanding().std()).fillna(0.0)

        group["baseline_mean"] = np.round(b_mean, 2)
        group["baseline_std"] = np.round(b_std, 2)

        group["upper_threshold"] = np.round(b_mean + upper_mult * b_std, 2)
        group["critical_threshold"] = np.round(b_mean + crit_mult * b_std, 2)

        # Safe division for std == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = np.where(b_std > 1e-6, (cases - b_mean) / b_std, 0.0)
        group["baseline_deviation"] = np.round(dev, 2)

        # Map to continuous 0-100 score: 0σ -> 0, 2σ -> 50, 4σ+ -> 100
        group["baseline_anomaly_score"] = np.clip(group["baseline_deviation"] * 25.0, 0.0, 100.0).round(2)

        # Categorical classification
        conditions = [
            group["baseline_deviation"] < upper_mult,
            (group["baseline_deviation"] >= upper_mult) & (group["baseline_deviation"] < crit_mult),
            group["baseline_deviation"] >= crit_mult,
        ]
        choices = ["normal", "potential_anomaly", "strong_anomaly"]
        group["statistical_anomaly"] = np.select(conditions, choices, default="normal")

        return group

    groups = []
    for _, g in out_df.groupby(["region", "disease"]):
        groups.append(_calc_group_baseline(g))
    out_df = pd.concat(groups, axis=0).reset_index(drop=True)


    counts = out_df["statistical_anomaly"].value_counts()
    logger.info(
        f"Level 1 Detection Complete — Normal: {counts.get('normal', 0)} | "
        f"Potential Anomaly: {counts.get('potential_anomaly', 0)} | "
        f"Strong Anomaly: {counts.get('strong_anomaly', 0)}"
    )

    return out_df
