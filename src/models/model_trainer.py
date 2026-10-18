"""EpiWatch Phase 3 — ML Model Trainer.

Trains three outbreak-detection classifiers on the prepared ML dataset:
  1. Logistic Regression   — Interpretable baseline
  2. Random Forest          — Ensemble tree-based model (main model)
  3. Gradient Boosting      — Boosted ensemble (strongest model)

Key Design Decisions:
---------------------
- Class imbalance is handled with SMOTE (applied only on training data).
- Features are standardised via StandardScaler (fitted on train, applied to test).
- All hyperparameters are configurable via config.yaml (ml_settings.models.*).
- The trained scaler is saved alongside each model for inference consistency.
- Models are persisted as .pkl files under models/ using joblib.
- Per-model metadata (train time, cv_scores) is saved as JSON.

SMOTE Note:
-----------
Synthetic Minority Over-sampling Technique is used solely to balance the training
class distribution. It operates in feature space, not by duplicating real records.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("model_trainer")

# ── Feature column sentinel (determined at runtime from dataset) ──────────────
_META_COLS = {"date", "region", "disease", "data_status"}


def _get_feature_and_target_cols(
    df: pd.DataFrame, target_col: str
) -> Tuple[List[str], str]:
    """Extract feature and target column names from a DataFrame."""
    exclude = _META_COLS | {target_col}
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols, target_col


def _build_model(model_name: str, config: Any) -> Optional[Any]:
    """Instantiate a sklearn model from config settings."""
    model_cfg = config.get(f"ml_settings.models.{model_name}", {})
    if not model_cfg.get("enabled", True):
        return None

    rs = int(config.get("ml_settings.random_state", 42))

    if model_name == "logistic_regression":
        return LogisticRegression(
            C=float(model_cfg.get("C", 1.0)),
            max_iter=int(model_cfg.get("max_iter", 1000)),
            random_state=rs,
            class_weight="balanced",   # fallback if SMOTE disabled
            solver="lbfgs",
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 100)),
            max_depth=int(model_cfg.get("max_depth", 10)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 5)),
            random_state=rs,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 100)),
            learning_rate=float(model_cfg.get("learning_rate", 0.1)),
            max_depth=int(model_cfg.get("max_depth", 5)),
            random_state=rs,
        )
    else:
        logger.warning(f"Unknown model name in config: {model_name}")
        return None


def train_models(
    train_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> Dict[str, Dict[str, Any]]:
    """Train all enabled classifiers on the training dataset.

    Parameters
    ----------
    train_df    : ML training DataFrame (features + target + meta columns)
    config_path : Path to config.yaml

    Returns
    -------
    results : Dict mapping model_name -> {model, scaler, cv_scores, metadata}
    """
    config = get_config(config_path)
    models_dir = config.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    target_col = config.get("ml_settings.target_column", "potential_anomaly")
    feature_cols, target_col = _get_feature_and_target_cols(train_df, target_col)

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df[target_col].values.astype(int)

    logger.info(f"Training features ({len(feature_cols)}): {feature_cols}")
    logger.info(
        f"Training label distribution — 0: {(y_train == 0).sum()}, "
        f"1: {(y_train == 1).sum()}"
    )

    # ── SMOTE for class balancing (applied only to train set) ───────────────
    use_smote = bool(config.get("ml_settings.use_smote", True))
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=int(config.get("ml_settings.random_state", 42)))
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            logger.info(
                f"SMOTE applied — resampled: 0: {(y_resampled == 0).sum()}, "
                f"1: {(y_resampled == 1).sum()}"
            )
        except ImportError:
            logger.warning("imbalanced-learn not installed; skipping SMOTE.")
            X_resampled, y_resampled = X_train, y_train
    else:
        X_resampled, y_resampled = X_train, y_train

    # ── Feature scaling (fit on SMOTE-resampled train) ──────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    # Save scaler (shared across all linear models; tree models also use it for
    # consistency in the inference pipeline)
    scaler_path = models_dir / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Feature scaler saved: {scaler_path}")

    # Save feature column order for inference
    feature_meta = {"feature_columns": feature_cols, "target_column": target_col}
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # ── Train each model ─────────────────────────────────────────────────────
    cv_folds = int(config.get("ml_settings.cv_folds", 5))
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=False)

    model_names = ["logistic_regression", "random_forest", "gradient_boosting"]
    results: Dict[str, Dict[str, Any]] = {}

    for model_name in model_names:
        model = _build_model(model_name, config)
        if model is None:
            logger.info(f"Skipping disabled model: {model_name}")
            continue

        logger.info(f"Training {model_name}...")
        t0 = time.time()

        # Cross-validation on SMOTE data (ROC-AUC)
        cv_scores = cross_val_score(
            model, X_scaled, y_resampled,
            cv=skf, scoring="roc_auc", n_jobs=-1
        )

        # Final fit on full resampled training set
        model.fit(X_scaled, y_resampled)
        elapsed = round(time.time() - t0, 2)

        logger.info(
            f"  {model_name} — CV ROC-AUC: "
            f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f} | "
            f"time: {elapsed}s"
        )

        # Persist model
        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)

        meta = {
            "model_name": model_name,
            "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
            "cv_roc_auc_std": round(float(cv_scores.std()), 4),
            "cv_roc_auc_scores": [round(float(s), 4) for s in cv_scores],
            "train_samples": int(len(X_resampled)),
            "n_features": int(len(feature_cols)),
            "training_time_seconds": elapsed,
            "smote_applied": use_smote,
            "label_source": "Phase2_statistical_rolling_threshold",
        }
        with open(models_dir / f"{model_name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        results[model_name] = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "cv_scores": cv_scores,
            "metadata": meta,
        }

    logger.info(f"Training complete. {len(results)} models saved to {models_dir}")
    return results
