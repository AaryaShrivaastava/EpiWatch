"""EpiWatch Phase 3 — Outbreak Risk Scorer.

Uses the best-performing trained model (by CV ROC-AUC) to score the full
labelled dataset and assign an interpretable outbreak risk level to each row.

Risk Level Mapping (probability of potential_anomaly == 1):
-----------------------------------------------------------
  LOW        : prob < 0.30
  MODERATE   : 0.30 <= prob < 0.50
  HIGH       : 0.50 <= prob < 0.70
  VERY HIGH  : prob >= 0.70

Output:
-------
- data/ml/outbreak_risk_scores.csv  — full dataset with risk probabilities and levels
- reports/ml/top_risk_windows.csv   — top-N highest-risk date/disease windows
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("risk_scorer")


# ── Risk threshold mapping ────────────────────────────────────────────────────
_RISK_THRESHOLDS = [
    (0.70, "VERY HIGH"),
    (0.50, "HIGH"),
    (0.30, "MODERATE"),
    (0.00, "LOW"),
]


def _assign_risk_level(prob: float) -> str:
    for threshold, level in _RISK_THRESHOLDS:
        if prob >= threshold:
            return level
    return "LOW"


def score_outbreak_risk(
    full_ml_df: pd.DataFrame,
    training_results: Dict[str, Dict[str, Any]],
    config_path: str = "config.yaml",
    top_n_windows: int = 30,
) -> pd.DataFrame:
    """Score all rows in the full ML dataset with outbreak risk probabilities.

    Uses the best model (highest CV ROC-AUC) from the training results.

    Parameters
    ----------
    full_ml_df       : The full ML dataset (features + target + meta).
    training_results : Output of model_trainer.train_models()
    config_path      : Path to config.yaml
    top_n_windows    : Number of top-risk rows to save as highlights.

    Returns
    -------
    risk_df : Full dataset with added risk_probability and risk_level columns.
    """
    config = get_config(config_path)
    ml_dir = Path("data/ml")
    ml_dir.mkdir(parents=True, exist_ok=True)
    ml_reports_dir = config.ml_reports_dir
    ml_reports_dir.mkdir(parents=True, exist_ok=True)

    target_col = config.get("ml_settings.target_column", "potential_anomaly")

    # ── Pick best model by CV ROC-AUC ────────────────────────────────────────
    best_model_name = max(
        training_results,
        key=lambda k: training_results[k]["metadata"]["cv_roc_auc_mean"]
    )
    best = training_results[best_model_name]
    model = best["model"]
    scaler = best["scaler"]
    feature_cols: List[str] = best["feature_cols"]

    logger.info(
        f"Scoring with best model: {best_model_name} "
        f"(CV ROC-AUC = {best['metadata']['cv_roc_auc_mean']})"
    )

    # ── Prepare features ─────────────────────────────────────────────────────
    X_full = full_ml_df[feature_cols].values.astype(float)
    # Fill any NaN (should be negligible after builder imputation)
    X_full = np.nan_to_num(X_full, nan=0.0)
    X_scaled = scaler.transform(X_full)

    # ── Score ─────────────────────────────────────────────────────────────────
    if hasattr(model, "predict_proba"):
        risk_probs = model.predict_proba(X_scaled)[:, 1]
    else:
        raw_scores = model.decision_function(X_scaled)
        # Normalise decision scores to [0, 1] using sigmoid
        risk_probs = 1.0 / (1.0 + np.exp(-raw_scores))

    risk_df = full_ml_df.copy()
    risk_df["risk_probability"] = np.round(risk_probs, 4)
    risk_df["risk_level"] = risk_df["risk_probability"].apply(_assign_risk_level)
    risk_df["best_model_used"] = best_model_name

    # ── Summary stats ─────────────────────────────────────────────────────────
    level_counts = risk_df["risk_level"].value_counts()
    logger.info(f"Risk level distribution:\n{level_counts.to_string()}")

    # ── Save full risk scores ─────────────────────────────────────────────────
    risk_path = ml_dir / "outbreak_risk_scores.csv"
    risk_df.to_csv(risk_path, index=False)
    logger.info(f"Full risk scores saved: {risk_path}")

    # ── Top-N risk windows ────────────────────────────────────────────────────
    top_cols = ["date", "region", "disease", "risk_probability", "risk_level",
                target_col, "new_cases", "cases_7d_avg", "growth_rate_7d"]
    available_top_cols = [c for c in top_cols if c in risk_df.columns]
    top_risk = (
        risk_df[available_top_cols]
        .sort_values("risk_probability", ascending=False)
        .head(top_n_windows)
    )
    top_path = ml_reports_dir / "top_risk_windows.csv"
    top_risk.to_csv(top_path, index=False)
    logger.info(f"Top {top_n_windows} risk windows saved: {top_path}")

    # ── Risk timeline CSV (for future dashboard / Phase 4 use) ───────────────
    timeline_cols = ["date", "region", "disease", "new_cases",
                     "risk_probability", "risk_level", target_col]
    available_timeline_cols = [c for c in timeline_cols if c in risk_df.columns]
    risk_timeline = risk_df[available_timeline_cols].copy()
    risk_timeline.to_csv(ml_dir / "risk_timeline.csv", index=False)
    logger.info("Risk timeline saved: data/ml/risk_timeline.csv")

    return risk_df
