"""EpiWatch Phase 3 — Level 3: Supervised Outbreak Risk Classifier.

Trains a supervised RandomForestClassifier (with class_weight="balanced")
and optional CalibratedClassifierCV on historical training data to predict `potential_outbreak`.

Key Outputs:
------------
- outbreak_probability: model-estimated probability of an emerging outbreak (0.0 to 1.0)
- feature_importance: feature ranking saved for downstream explainability modules

Disclaimer:
-----------
`outbreak_probability` is a model-estimated statistical probability, NOT a clinical
or medically confirmed diagnosis probability.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("outbreak_classifier")


def train_outbreak_classifier(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    config_path: str = "config.yaml",
) -> Tuple[RandomForestClassifier, Any, pd.DataFrame, Dict[str, Any]]:
    """Train supervised Random Forest classifier and probability calibrator.

    Parameters
    ----------
    train_df     : Training dataset containing target 'potential_outbreak'
    feature_cols : List of feature names
    config_path  : Path to config.yaml

    Returns
    -------
    model, calibrated_model, feature_importance_df, metadata
    """
    config = get_config(config_path)
    outbreak_models_dir = config.outbreak_models_dir
    outbreak_reports_dir = config.outbreak_reports_dir
    outbreak_models_dir.mkdir(parents=True, exist_ok=True)
    outbreak_reports_dir.mkdir(parents=True, exist_ok=True)

    n_est = int(config.get("outbreak_detection.classifier.n_estimators", 100))
    max_d = int(config.get("outbreak_detection.classifier.max_depth", 10))
    rs = int(config.get("outbreak_detection.classifier.random_state", 42))

    target_col = "potential_outbreak"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' missing from training DataFrame.")

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df[target_col].values.astype(int)

    logger.info(
        f"Training Supervised Outbreak Classifier (RF n_estimators={n_est}, "
        f"max_depth={max_d}, class_weight=balanced)..."
    )
    logger.info(f"Train label distribution — 0: {(y_train == 0).sum()}, 1: {(y_train == 1).sum()}")

    base_model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=max_d,
        class_weight="balanced",
        random_state=rs,
        n_jobs=-1,
    )
    base_model.fit(X_train, y_train)

    # Probability Calibration using 3-fold CV
    logger.info("Calibrating classifier probabilities with CalibratedClassifierCV...")
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)

    # Feature Importance
    importances = base_model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi_df["rank"] = fi_df.index + 1
    fi_df.to_csv(outbreak_reports_dir / "feature_importance.csv", index=False)
    logger.info(f"Feature importance saved to {outbreak_reports_dir / 'feature_importance.csv'}")

    # Save joblib models
    joblib.dump(base_model, outbreak_models_dir / "outbreak_classifier.joblib")
    joblib.dump(calibrated_model, outbreak_models_dir / "calibrated_classifier.joblib")

    metadata = {
        "model_type": "RandomForestClassifier",
        "calibration": "CalibratedClassifierCV(sigmoid)",
        "n_estimators": n_est,
        "max_depth": max_d,
        "class_weight": "balanced",
        "random_seed": rs,
        "n_train_samples": int(len(train_df)),
        "target_column": target_col,
        "label_type": "weak_statistical_label",
        "version": "0.3.0",
    }
    with open(outbreak_models_dir / "classifier_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return base_model, calibrated_model, fi_df, metadata


def predict_outbreak_probability(
    df: pd.DataFrame,
    feature_cols: List[str],
    calibrated_model: Any,
) -> pd.DataFrame:
    """Predict model-estimated outbreak probabilities.

    Returns
    -------
    out_df with added outbreak_probability column (0.00 to 1.00)
    """
    out_df = df.copy()
    X = df[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    probs = calibrated_model.predict_proba(X)[:, 1]
    out_df["outbreak_probability"] = np.round(probs, 4)

    return out_df
