"""EpiWatch Phase 3 — ML Dataset Builder.

Constructs the supervised ML dataset for outbreak detection by:
1. Loading the master feature dataset (Phase 1 output).
2. Loading Phase 2 statistical anomaly labels.
3. Joining labels onto features.
4. Selecting the feature columns defined in config.yaml.
5. Performing median imputation for any residual NaN values.
6. Performing chronological train/test split (no random shuffling).

Important:
----------
- The target label ``potential_anomaly`` is derived from Phase 2 statistical
  rolling-threshold analysis. It is a STATISTICAL PROXY for outbreak risk,
  NOT a medically confirmed outbreak label.
- The chronological split ensures no future information leaks into training.
- Data provenance (SYNTHETIC / SOURCE_CALIBRATED_SYNTHETIC) is preserved in
  the output metadata but NOT used as a model feature.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("dataset_builder")


def build_ml_dataset(
    config_path: str = "config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ML-ready train, test, and full labelled datasets.

    Steps
    -----
    1. Load master features (data/features/epiwatch_master_features.csv).
    2. Load Phase 2 anomaly labels (data/analytics/potential_anomalies.csv).
    3. Join labels → features on (date, region, disease).
    4. Select feature columns from config; drop rows with no label.
    5. Median-impute any NaN feature values.
    6. Chronological train/test split by date.
    7. Save outputs to data/ml/.

    Returns
    -------
    train_df : pd.DataFrame  — Training set (features + target)
    test_df  : pd.DataFrame  — Test set (features + target)
    full_df  : pd.DataFrame  — Full labelled dataset
    """
    config = get_config(config_path)

    features_dir = config.features_dir
    analytics_dir = config.analytics_data_dir
    ml_dir = Path("data/ml")
    ml_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load datasets ────────────────────────────────────────────────────
    logger.info("Loading master feature dataset...")
    master_path = features_dir / "epiwatch_master_features.csv"
    if not master_path.exists():
        raise FileNotFoundError(
            f"Master features not found: {master_path}. "
            "Run Phase 1 pipeline first."
        )
    master_df = pd.read_csv(master_path, parse_dates=["date"], low_memory=False)

    logger.info("Loading Phase 2 anomaly labels...")
    anom_path = analytics_dir / "potential_anomalies.csv"
    if not anom_path.exists():
        raise FileNotFoundError(
            f"Anomaly labels not found: {anom_path}. "
            "Run Phase 2 (EDA) pipeline first."
        )
    anom_df = pd.read_csv(anom_path, parse_dates=["date"], low_memory=False)

    # ── 2. Extract labels ────────────────────────────────────────────────────
    label_cols = ["date", "region", "disease", "potential_anomaly",
                  "baseline_mean", "upper_threshold"]
    available_label_cols = [c for c in label_cols if c in anom_df.columns]
    anom_labels = anom_df[available_label_cols].copy()

    # ── 3. Join labels onto master features ─────────────────────────────────
    logger.info("Joining anomaly labels onto master feature dataset...")
    join_keys = ["date", "region", "disease"]
    full_df = master_df.merge(anom_labels, on=join_keys, how="left")
    logger.info(f"Joined dataset shape: {full_df.shape}")

    # ── 4. Feature & target selection ───────────────────────────────────────
    target_col = config.get("ml_settings.target_column", "potential_anomaly")
    feature_cols_cfg = config.get("ml_settings.feature_columns", [])

    # Keep only features actually present in the joined dataframe
    feature_cols = [c for c in feature_cols_cfg if c in full_df.columns]
    missing_cfg = set(feature_cols_cfg) - set(feature_cols)
    if missing_cfg:
        logger.warning(f"Config features not found in dataset (skipped): {missing_cfg}")

    logger.info(f"Selected {len(feature_cols)} features: {feature_cols}")

    # Drop rows with no label
    full_df = full_df.dropna(subset=[target_col]).copy()
    full_df[target_col] = full_df[target_col].astype(int)

    logger.info(
        f"Label distribution:\n"
        f"  Non-anomaly (0): {(full_df[target_col] == 0).sum()}\n"
        f"  Potential anomaly (1): {(full_df[target_col] == 1).sum()}"
    )

    # ── 5. Impute residual NaNs with column median ───────────────────────────
    logger.info("Imputing residual NaN values with column medians...")
    feature_df = full_df[feature_cols].copy()
    col_medians: Dict[str, float] = {}
    for col in feature_cols:
        median_val = feature_df[col].median()
        col_medians[col] = float(median_val)
        feature_df[col] = feature_df[col].fillna(median_val)

    # Attach imputation metadata for audit
    imputation_df = pd.DataFrame(
        list(col_medians.items()), columns=["feature", "imputed_median"]
    )
    imputation_df.to_csv(ml_dir / "feature_imputation_medians.csv", index=False)

    # ── 6. Chronological train/test split ───────────────────────────────────
    test_size = float(config.get("ml_settings.test_size", 0.2))
    full_df = full_df.sort_values("date").reset_index(drop=True)
    feature_df = feature_df.loc[full_df.index].reset_index(drop=True)

    split_idx = int(len(full_df) * (1 - test_size))
    cutoff_date = full_df["date"].iloc[split_idx]

    logger.info(
        f"Chronological split: train=[start, {cutoff_date.date()}), "
        f"test=[{cutoff_date.date()}, end]"
    )

    # NOTE: For this capstone dataset, temporal evaluation also uses
    # stratified k-fold CV to produce meaningful generalisation metrics,
    # since the synthetic data's anomaly patterns may shift across years.
    # The chronological split is preserved for realistic deployment simulation.

    # Build combined feature+target DataFrames for output
    ml_cols = feature_cols + [target_col]
    meta_cols = ["date", "region", "disease", "data_status"] if "data_status" in full_df.columns else ["date", "region", "disease"]

    full_ml_df = pd.concat(
        [full_df[meta_cols].reset_index(drop=True),
         feature_df,
         full_df[[target_col]].reset_index(drop=True)],
        axis=1
    )

    train_df = full_ml_df.iloc[:split_idx].reset_index(drop=True)
    test_df = full_ml_df.iloc[split_idx:].reset_index(drop=True)

    logger.info(f"Train set: {len(train_df)} samples | Test set: {len(test_df)} samples")

    # ── 7. Save ─────────────────────────────────────────────────────────────
    train_df.to_csv(ml_dir / "train_dataset.csv", index=False)
    test_df.to_csv(ml_dir / "test_dataset.csv", index=False)
    full_ml_df.to_csv(ml_dir / "full_ml_dataset.csv", index=False)

    # Split metadata
    split_meta = pd.DataFrame([{
        "total_samples": len(full_ml_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "cutoff_date": str(cutoff_date.date()),
        "n_features": len(feature_cols),
        "target_column": target_col,
        "train_positive_rate": round(train_df[target_col].mean(), 4),
        "test_positive_rate": round(test_df[target_col].mean(), 4),
        "split_method": "chronological",
        "label_source": "Phase2_statistical_rolling_threshold",
    }])
    split_meta.to_csv(ml_dir / "split_metadata.csv", index=False)
    logger.info("ML datasets saved to data/ml/")

    return train_df, test_df, full_ml_df
