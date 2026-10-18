"""EpiWatch Phase 3 — ML Outbreak Detection Pipeline Orchestrator.

Pipeline Flow:
    Phase 1 + Phase 2 Artifacts
        → Dataset Builder   (build supervised ML dataset + chronological split)
        → Model Trainer     (train LR, RF, GB with SMOTE + cross-validation)
        → Model Evaluator   (test-set metrics: ROC-AUC, F1, PR curve, confusion matrix)
        → Risk Scorer       (score full dataset, assign risk levels)
        → ML Visualisations (comparison ROC/PR curves, risk timeline, distribution)
        → ML Reports        (classification reports, model comparison CSV, summary MD)

Prerequisites:
--------------
    Run Phase 1 (data_pipeline.py) and Phase 2 (eda_pipeline.py) first.
    Required files:
        - data/features/epiwatch_master_features.csv
        - data/analytics/potential_anomalies.csv
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.models.dataset_builder import build_ml_dataset
from src.models.model_trainer import train_models
from src.models.model_evaluator import evaluate_models, cross_validate_full_dataset
from src.models.risk_scorer import score_outbreak_risk
from src.models.ml_plots import plot_ml_visualizations
from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("ml_pipeline")


def run_ml_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Execute the complete Phase 3 ML Outbreak Detection Pipeline.

    Returns
    -------
    Dict with keys:
        train_df        : pd.DataFrame — ML training set
        test_df         : pd.DataFrame — ML test set
        full_ml_df      : pd.DataFrame — Full labelled ML dataset
        training_results: Dict         — Model objects + metadata
        comparison_df   : pd.DataFrame — Model evaluation comparison
        risk_df         : pd.DataFrame — Full dataset with risk scores
    """
    logger.info("=" * 60)
    logger.info("EpiWatch Phase 3 — ML Outbreak Detection Pipeline")
    logger.info("=" * 60)

    config = get_config(config_path)

    # ── Step 1: Build ML Dataset ─────────────────────────────────────────────
    logger.info("[Step 1/5] Building ML dataset...")
    train_df, test_df, full_ml_df = build_ml_dataset(config_path=config_path)
    logger.info(
        f"Dataset ready — Train: {len(train_df)} | Test: {len(test_df)} | "
        f"Total: {len(full_ml_df)}"
    )

    # ── Step 2: Train Models ─────────────────────────────────────────────────
    logger.info("[Step 2/5] Training outbreak detection models...")
    training_results = train_models(train_df, config_path=config_path)
    logger.info(f"Trained {len(training_results)} model(s): {list(training_results.keys())}")

    # ── Step 3: Evaluate Models ──────────────────────────────────────────────
    logger.info("[Step 3/5] Evaluating models (test set + cross-validation)...")
    comparison_df = evaluate_models(test_df, training_results, config_path=config_path)
    cv_summary_df = cross_validate_full_dataset(full_ml_df, training_results, config_path=config_path)
    best_model = cv_summary_df.iloc[0]["model"]
    best_auc = cv_summary_df.iloc[0]["cv_roc_auc_mean"]
    logger.info(f"Best model by Stratified CV: {best_model} (CV ROC-AUC={best_auc})")


    # ── Step 4: Score Full Dataset ───────────────────────────────────────────
    logger.info("[Step 4/5] Scoring outbreak risk across full dataset...")
    risk_df = score_outbreak_risk(
        full_ml_df=full_ml_df,
        training_results=training_results,
        config_path=config_path,
    )
    logger.info(
        f"Risk scoring complete — "
        f"VERY HIGH: {(risk_df['risk_level'] == 'VERY HIGH').sum()} | "
        f"HIGH: {(risk_df['risk_level'] == 'HIGH').sum()} | "
        f"MODERATE: {(risk_df['risk_level'] == 'MODERATE').sum()} | "
        f"LOW: {(risk_df['risk_level'] == 'LOW').sum()}"
    )

    # ── Step 5: Generate Visualisations ─────────────────────────────────────
    logger.info("[Step 5/5] Generating Phase 3 ML visualisations...")
    plot_ml_visualizations(
        test_df=test_df,
        training_results=training_results,
        risk_df=risk_df,
        config_path=config_path,
    )

    logger.info("=" * 60)
    logger.info("Phase 3 ML Pipeline Completed Successfully!")
    logger.info(f"  Models saved    : {config.models_dir}")
    logger.info(f"  ML reports      : {config.ml_reports_dir}")
    logger.info(f"  ML figures      : {config.figures_dir / 'ml'}")
    logger.info(f"  Risk scores     : data/ml/outbreak_risk_scores.csv")
    logger.info("=" * 60)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "full_ml_df": full_ml_df,
        "training_results": training_results,
        "comparison_df": comparison_df,
        "risk_df": risk_df,
    }


if __name__ == "__main__":
    results = run_ml_pipeline()
    print("\n=== Phase 3 Complete ===")
    print(f"Models trained: {list(results['training_results'].keys())}")
    print("\nModel comparison:")
    print(results["comparison_df"][
        ["model", "roc_auc_test", "avg_precision_test", "f1_score", "recall"]
    ].to_string(index=False))
