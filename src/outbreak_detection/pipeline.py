"""EpiWatch Phase 3 — Early Outbreak Detection Main Pipeline Orchestrator.

Executes the complete multi-level outbreak detection workflow:
  1. Feature Builder & Chronological Split
  2. Level 1 — Historical Baseline Detector
  3. Level 2 — Multi-Signal Statistical Detector
  4. Level 2 — Unsupervised Isolation Forest
  5. Level 3 — Supervised Outbreak Risk Classifier
  6. Signal Aggregator & Outbreak Alert Status
  7. Outbreak Severity & Deterministic Contributing Signals
  8. Evaluation, Lead Time Calculation, Visualisations, and Reports
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.outbreak_detection.feature_builder import load_and_build_features
from src.outbreak_detection.baseline_detector import detect_baseline_anomalies
from src.outbreak_detection.statistical_detector import compute_statistical_signals
from src.outbreak_detection.isolation_forest import train_isolation_forest, predict_isolation_forest
from src.outbreak_detection.outbreak_classifier import train_outbreak_classifier, predict_outbreak_probability
from src.outbreak_detection.signal_aggregator import aggregate_signals
from src.outbreak_detection.severity import assign_severity_and_reasons
from src.evaluation.outbreak_evaluation import (
    evaluate_detection_models,
    group_outbreak_episodes,
    generate_outbreak_visualizations,
    write_evaluation_report,
)
from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("outbreak_pipeline")


def run_outbreak_detection_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Execute the complete Phase 3 Early Outbreak Detection Pipeline.

    Returns
    -------
    Dict containing output DataFrames, evaluation metrics, and episode summaries
    """
    logger.info("=" * 70)
    logger.info("EpiWatch Phase 3 — Early Outbreak Detection Pipeline Starting")
    logger.info("=" * 70)

    config = get_config(config_path)

    # ── 1. Feature Building & Chronological Split ───────────────────────────
    logger.info("[Step 1/8] Loading features and performing chronological split...")
    full_df, train_df, val_df, test_df, feature_cols = load_and_build_features(config_path)

    # ── 2. Level 1 Historical Baseline Detector ───────────────────────────────
    logger.info("[Step 2/8] Executing Level 1 Historical Baseline Detector...")
    df_lvl1 = detect_baseline_anomalies(full_df, config_path)

    # ── 3. Level 2 Statistical Signals ────────────────────────────────────────
    logger.info("[Step 3/8] Executing Level 2 Multi-Dimensional Statistical Detector...")
    df_lvl2_stat = compute_statistical_signals(df_lvl1, config_path)

    # ── 4. Level 2 Isolation Forest (Train on train_df only) ──────────────────
    logger.info("[Step 4/8] Training Level 2 Isolation Forest on training split...")
    iso_model, iso_scaler, iso_meta = train_isolation_forest(train_df, feature_cols, config_path)
    df_lvl2_iso = predict_isolation_forest(df_lvl2_stat, feature_cols, iso_model, iso_scaler)

    # ── 5. Level 3 Supervised Outbreak Classifier (Train on train_df only) ───
    logger.info("[Step 5/8] Training Level 3 Supervised Outbreak Classifier...")
    clf_base, clf_calibrated, fi_df, clf_meta = train_outbreak_classifier(train_df, feature_cols, config_path)
    df_lvl3 = predict_outbreak_probability(df_lvl2_iso, feature_cols, clf_calibrated)

    # ── 6. Signal Aggregator & Outbreak Status ────────────────────────────────
    logger.info("[Step 6/8] Aggregating signals and assigning Outbreak Status...")
    df_aggregated = aggregate_signals(df_lvl3, config_path)

    # ── 7. Severity & Contributing Signals ───────────────────────────────────
    logger.info("[Step 7/8] Computing Severity Tiers and Contributing Signals...")
    final_df = assign_severity_and_reasons(df_aggregated)

    # Save output analytics dataset
    analytics_dir = config.analytics_data_dir
    analytics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = analytics_dir / "outbreak_detection_results.csv"
    final_df.to_csv(out_csv, index=False)
    logger.info(f"Outbreak detection results saved to {out_csv}")

    # ── 8. Evaluation, Lead Time, Visualisations & Reports ───────────────────
    logger.info("[Step 8/8] Running Evaluation, Episode Grouping, and Visualization...")

    # Filter test set results for evaluation
    test_dates = test_df["date"] if "date" in test_df.columns else pd.Series()
    eval_test_df = final_df[final_df["date"].isin(test_dates)].copy()

    comparison_df, metrics_dict = evaluate_detection_models(eval_test_df, config_path)
    episodes_df = group_outbreak_episodes(final_df, config_path)
    generate_outbreak_visualizations(final_df, eval_test_df, episodes_df, config_path)
    write_evaluation_report(comparison_df, episodes_df, config_path)

    logger.info("=" * 70)
    logger.info("EpiWatch Phase 3 Early Outbreak Detection Pipeline Completed Successfully!")
    logger.info(f"  Results dataset: {out_csv}")
    logger.info(f"  Reports        : {config.outbreak_reports_dir}")
    logger.info(f"  Figures        : {config.figures_dir / 'outbreak_detection'}")
    logger.info(f"  Models         : {config.outbreak_models_dir}")
    logger.info("=" * 70)

    return {
        "results_df": final_df,
        "train_df": train_df,
        "test_df": eval_test_df,
        "comparison_df": comparison_df,
        "episodes_df": episodes_df,
        "metrics": metrics_dict,
    }


if __name__ == "__main__":
    results = run_outbreak_detection_pipeline()
    print("\n=== Phase 3 Early Outbreak Detection Pipeline Complete ===")
    print("Identified Episodes:", len(results["episodes_df"]))
    if not results["comparison_df"].empty:
        print("\nModel Comparison Metrics:")
        print(results["comparison_df"][["model", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
