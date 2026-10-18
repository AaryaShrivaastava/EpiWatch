"""EpiWatch Phase 3 — Signal Aggregator & Outbreak Status Assigner.

Aggregates Level 1, Level 2, and Level 3 signals into a unified `overall_outbreak_score` (0–100)
and maps it to an interpretable `outbreak_status`:

Signal Weights (Engineering Defaults):
--------------------------------------
  - Statistical Baseline Anomaly Score : 25%
  - Isolation Forest Anomaly Score     : 20%
  - Classifier Outbreak Probability    : 35%
  - Case Growth Signal                 : 10%
  - Incidence Rate Signal              : 10%

Outbreak Status Tiers:
----------------------
  -  0 – 24 : NORMAL
  - 25 – 49 : WATCH
  - 50 – 74 : ELEVATED
  - 75 – 100: HIGH_ALERT

Disclaimer:
-----------
These alert tiers are system operational warning categories, NOT clinical diagnostic thresholds.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("signal_aggregator")


def aggregate_signals(
    df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """Aggregate multi-level signals into overall_outbreak_score and outbreak_status.

    Parameters
    ----------
    df          : DataFrame with baseline_anomaly_score, anomaly_score_0_100, outbreak_probability, etc.
    config_path : Path to config.yaml

    Returns
    -------
    out_df : DataFrame with overall_outbreak_score and outbreak_status
    """
    config = get_config(config_path)

    w_stat = float(config.get("outbreak_detection.signal_weights.statistical_anomaly", 0.25))
    w_iso = float(config.get("outbreak_detection.signal_weights.isolation_anomaly", 0.20))
    w_clf = float(config.get("outbreak_detection.signal_weights.classifier_probability", 0.35))
    w_growth = float(config.get("outbreak_detection.signal_weights.growth_signal", 0.10))
    w_inc = float(config.get("outbreak_detection.signal_weights.incidence_signal", 0.10))

    normal_max = float(config.get("outbreak_detection.status_thresholds.normal_max", 24))
    watch_max = float(config.get("outbreak_detection.status_thresholds.watch_max", 49))
    elevated_max = float(config.get("outbreak_detection.status_thresholds.elevated_max", 74))

    logger.info(
        f"Aggregating signals (Weights: Stat={w_stat}, Iso={w_iso}, Clf={w_clf}, "
        f"Growth={w_growth}, Inc={w_inc})..."
    )

    out_df = df.copy()

    # Extract component scores (0–100 scale)
    stat_score = out_df.get("baseline_anomaly_score", pd.Series(0.0, index=out_df.index)).fillna(0.0)
    iso_score = out_df.get("anomaly_score_0_100", pd.Series(0.0, index=out_df.index)).fillna(0.0)
    clf_score = (out_df.get("outbreak_probability", pd.Series(0.0, index=out_df.index)).fillna(0.0)) * 100.0
    growth_score = (out_df.get("growth_signal", pd.Series(0.0, index=out_df.index)).fillna(0.0)) * 100.0
    inc_score = (out_df.get("incidence_signal", pd.Series(0.0, index=out_df.index)).fillna(0.0)) * 100.0

    raw_overall = (
        w_stat * stat_score +
        w_iso * iso_score +
        w_clf * clf_score +
        w_growth * growth_score +
        w_inc * inc_score
    )

    out_df["overall_outbreak_score"] = np.clip(np.round(raw_overall, 2), 0.0, 100.0)

    # Assign outbreak status
    conditions = [
        out_df["overall_outbreak_score"] <= normal_max,
        (out_df["overall_outbreak_score"] > normal_max) & (out_df["overall_outbreak_score"] <= watch_max),
        (out_df["overall_outbreak_score"] > watch_max) & (out_df["overall_outbreak_score"] <= elevated_max),
        out_df["overall_outbreak_score"] > elevated_max,
    ]
    choices = ["NORMAL", "WATCH", "ELEVATED", "HIGH_ALERT"]

    out_df["outbreak_status"] = np.select(conditions, choices, default="NORMAL")

    counts = out_df["outbreak_status"].value_counts()
    logger.info(
        f"Signal Aggregation Complete — NORMAL: {counts.get('NORMAL', 0)} | "
        f"WATCH: {counts.get('WATCH', 0)} | ELEVATED: {counts.get('ELEVATED', 0)} | "
        f"HIGH_ALERT: {counts.get('HIGH_ALERT', 0)}"
    )

    return out_df
