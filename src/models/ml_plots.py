"""EpiWatch Phase 3 — ML Model Visualisations.

Generates additional Phase 3-specific plots:
  - All three ROC curves on a single comparison chart
  - All three PR curves on a single comparison chart
  - Risk score distribution histogram
  - Risk probability timeline per disease
  - Risk level heatmap by month and disease
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("ml_plots")

_MODEL_COLORS = {
    "logistic_regression": "steelblue",
    "random_forest": "seagreen",
    "gradient_boosting": "darkorange",
}


def _scale_test(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    target_col: str,
) -> tuple:
    X = test_df[feature_cols].values.astype(float)
    y = test_df[target_col].values.astype(int)
    return scaler.transform(X), y


def plot_roc_comparison(
    test_df: pd.DataFrame,
    training_results: Dict[str, Dict[str, Any]],
    out_dir: Path,
    config_path: str = "config.yaml",
) -> None:
    """Plot all model ROC curves on a single figure."""
    config = get_config(config_path)
    target_col = config.get("ml_settings.target_column", "potential_anomaly")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline (AUC=0.50)")

    for model_name, res in training_results.items():
        X_test, y_test = _scale_test(
            test_df, res["feature_cols"], res["scaler"], target_col
        )
        y_prob = (
            res["model"].predict_proba(X_test)[:, 1]
            if hasattr(res["model"], "predict_proba")
            else res["model"].decision_function(X_test)
        )
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        label = f"{model_name.replace('_', ' ').title()} (AUC={auc:.4f})"
        ax.plot(fpr, tpr, lw=2, color=_MODEL_COLORS.get(model_name, "grey"), label=label)

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve Comparison — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "roc_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC comparison plot saved: {out_path}")


def plot_pr_comparison(
    test_df: pd.DataFrame,
    training_results: Dict[str, Dict[str, Any]],
    out_dir: Path,
    config_path: str = "config.yaml",
) -> None:
    """Plot all model Precision-Recall curves on a single figure."""
    config = get_config(config_path)
    target_col = config.get("ml_settings.target_column", "potential_anomaly")

    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, res in training_results.items():
        X_test, y_test = _scale_test(
            test_df, res["feature_cols"], res["scaler"], target_col
        )
        y_prob = (
            res["model"].predict_proba(X_test)[:, 1]
            if hasattr(res["model"], "predict_proba")
            else res["model"].decision_function(X_test)
        )
        ap = average_precision_score(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        label = f"{model_name.replace('_', ' ').title()} (AP={ap:.4f})"
        ax.plot(rec, prec, lw=2, color=_MODEL_COLORS.get(model_name, "grey"), label=label)

    baseline = test_df[target_col].mean() if target_col in test_df.columns else 0.3
    ax.axhline(y=baseline, linestyle="--", color="grey", lw=1,
               label=f"Baseline prevalence = {baseline:.3f}")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve Comparison — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "pr_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PR comparison plot saved: {out_path}")


def plot_risk_distribution(risk_df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of outbreak risk probabilities."""
    if "risk_probability" not in risk_df.columns:
        logger.warning("risk_probability column not found, skipping risk distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    axes[0].hist(risk_df["risk_probability"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    for threshold, level in [(0.70, "Very High"), (0.50, "High"), (0.30, "Moderate")]:
        axes[0].axvline(threshold, color="red", linestyle="--", lw=1.2, alpha=0.7,
                        label=f"{level}: {threshold}")
    axes[0].set_xlabel("Outbreak Risk Probability", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Distribution of Outbreak Risk Probabilities", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Pie chart of risk levels
    if "risk_level" in risk_df.columns:
        level_order = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
        level_colors = ["seagreen", "gold", "darkorange", "crimson"]
        counts = risk_df["risk_level"].value_counts()
        levels = [l for l in level_order if l in counts.index]
        vals = [counts[l] for l in levels]
        colors = [level_colors[level_order.index(l)] for l in levels]
        axes[1].pie(vals, labels=levels, colors=colors, autopct="%1.1f%%",
                    startangle=90, textprops={"fontsize": 10})
        axes[1].set_title("Risk Level Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / "risk_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Risk distribution plot saved: {out_path}")


def plot_risk_timeline(risk_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot risk probability over time per disease."""
    if "risk_probability" not in risk_df.columns or "date" not in risk_df.columns:
        return

    df = risk_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    diseases = df["disease"].unique() if "disease" in df.columns else ["All"]
    n = len(diseases)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, disease in zip(axes, diseases):
        sub = df[df["disease"] == disease].sort_values("date")
        ax.fill_between(sub["date"], sub["risk_probability"],
                        alpha=0.4, color="steelblue")
        ax.plot(sub["date"], sub["risk_probability"], lw=1.2, color="steelblue")

        # Threshold lines
        ax.axhline(0.70, color="crimson", linestyle="--", lw=1, label="Very High (0.70)")
        ax.axhline(0.50, color="darkorange", linestyle="--", lw=1, label="High (0.50)")
        ax.axhline(0.30, color="gold", linestyle="--", lw=1, label="Moderate (0.30)")

        ax.set_ylabel("Risk Probability", fontsize=10)
        ax.set_title(f"Outbreak Risk Timeline — {disease}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "risk_timeline.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Risk timeline plot saved: {out_path}")


def plot_ml_visualizations(
    test_df: pd.DataFrame,
    training_results: Dict[str, Dict[str, Any]],
    risk_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> None:
    """Run all Phase 3 ML visualisation plots."""
    config = get_config(config_path)
    figures_dir = config.figures_dir / "ml"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_comparison(test_df, training_results, figures_dir, config_path)
    plot_pr_comparison(test_df, training_results, figures_dir, config_path)
    plot_risk_distribution(risk_df, figures_dir)
    plot_risk_timeline(risk_df, figures_dir)

    logger.info("All Phase 3 ML visualisations completed.")
