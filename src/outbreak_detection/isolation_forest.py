"""EpiWatch Phase 3 — Level 2: Isolation Forest Anomaly Detector.

Fits an unsupervised Isolation Forest model (sklearn.ensemble.IsolationForest)
strictly on historical training data to detect multidimensional operational anomalies.

Key Rules:
----------
1. Model is fitted strictly on `train_df` (no future data leakage).
2. Features are standardized via StandardScaler fitted on train data.
3. Raw IsolationForest decision_function values are transformed into an interpretable
   `anomaly_score_0_100` where 0 = normal and 100 = extremely anomalous.
4. Model artifacts are saved to models/outbreak/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("isolation_forest")


def train_isolation_forest(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    config_path: str = "config.yaml",
) -> Tuple[IsolationForest, StandardScaler, Dict[str, Any]]:
    """Fit Isolation Forest model on training DataFrame.

    Parameters
    ----------
    train_df     : Training dataset
    feature_cols : List of feature names to use
    config_path  : Path to config.yaml

    Returns
    -------
    model, scaler, metadata
    """
    config = get_config(config_path)
    outbreak_models_dir = config.outbreak_models_dir
    outbreak_models_dir.mkdir(parents=True, exist_ok=True)

    contam = float(config.get("outbreak_detection.isolation_forest.contamination", 0.05))
    n_est = int(config.get("outbreak_detection.isolation_forest.n_estimators", 100))
    rs = int(config.get("outbreak_detection.isolation_forest.random_state", 42))

    logger.info(f"Training Isolation Forest (n_estimators={n_est}, contamination={contam}, seed={rs})...")

    X_train = train_df[feature_cols].values.astype(float)
    X_train = np.nan_to_num(X_train, nan=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=n_est,
        contamination=contam,
        random_state=rs,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Save artifacts
    joblib.dump(model, outbreak_models_dir / "isolation_forest.joblib")
    joblib.dump(scaler, outbreak_models_dir / "isolation_scaler.joblib")

    with open(outbreak_models_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols}, f, indent=2)

    metadata = {
        "model_type": "IsolationForest",
        "contamination": contam,
        "n_estimators": n_est,
        "random_seed": rs,
        "n_train_samples": int(len(train_df)),
        "train_start_date": str(train_df["date"].min().date()) if "date" in train_df.columns else None,
        "train_end_date": str(train_df["date"].max().date()) if "date" in train_df.columns else None,
        "features": feature_cols,
        "version": "0.3.0",
    }
    with open(outbreak_models_dir / "isolation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Isolation Forest model saved to {outbreak_models_dir}")
    return model, scaler, metadata


def predict_isolation_forest(
    df: pd.DataFrame,
    feature_cols: List[str],
    model: IsolationForest,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Predict Isolation Forest anomaly scores on a DataFrame.

    Transformation Formula:
    -----------------------
    Sklearn's decision_function returns negative values for anomalies and positive for inliers.
    We convert it to `anomaly_score_0_100`:
        score = clip((0.5 - decision_function) * 100, 0, 100)
    Where:
        - Typical inliers (decision_function ~ +0.2) -> score ~ 30 (Normal)
        - Boundary points (decision_function ~ 0.0)  -> score ~ 50 (Elevated)
        - Severe outliers (decision_function ~ -0.3) -> score ~ 80+ (High Anomaly)

    Returns
    -------
    out_df with isolation_score, anomaly_score_0_100, and isolation_anomaly columns
    """
    out_df = df.copy()
    X = df[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    X_scaled = scaler.transform(X)

    raw_decisions = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)  # -1 for anomaly, 1 for inlier

    out_df["isolation_score"] = np.round(raw_decisions, 4)
    # 0 = normal, 100 = extremely anomalous
    scores_0_100 = np.clip((0.25 - raw_decisions) * 125.0, 0.0, 100.0)
    out_df["anomaly_score_0_100"] = np.round(scores_0_100, 2)
    out_df["isolation_anomaly"] = (preds == -1).astype(int)

    return out_df
