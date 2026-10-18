"""EpiWatch Phase 3 — Outbreak Detection Evaluator, Episode Grouper & Reporter.

Comprehensive evaluation and analytics for Phase 3:
1. Multi-model Performance Metrics (Baseline, Isolation Forest, Supervised Classifier).
2. Outbreak Episode Grouping (aggregates consecutive alerts into continuous episode records).
3. Early Warning Lead Time Calculation (first detection date to episode peak date).
4. Automated Publication-Grade Visualisations (ROC, PR Curve, Confusion Matrix, Timelines).
5. Detailed Model Comparison CSVs and Markdown Evaluation Report.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("outbreak_evaluation")


# ── 1. Model Evaluation Metrics ──────────────────────────────────────────────

def evaluate_detection_models(
    test_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate Level 1 (Baseline), Level 2 (Isolation Forest), and Level 3 (Supervised Classifier).

    Target: `potential_outbreak` (weak statistical label).

    Returns
    -------
    comparison_df, metrics_dict
    """
    config = get_config(config_path)
    outbreak_reports_dir = config.outbreak_reports_dir
    outbreak_reports_dir.mkdir(parents=True, exist_ok=True)

    target_col = "potential_outbreak"
    if target_col not in test_df.columns:
        logger.warning(f"Target '{target_col}' missing from test set. Evaluation skipped.")
        return pd.DataFrame(), {}

    y_true = test_df[target_col].values.astype(int)

    models_to_eval = [
        ("Statistical Baseline (Level 1)", (test_df["statistical_anomaly"] != "normal").astype(int), test_df["baseline_anomaly_score"] / 100.0),
        ("Isolation Forest (Level 2)", test_df.get("isolation_anomaly", pd.Series(0, index=test_df.index)).astype(int), test_df.get("anomaly_score_0_100", pd.Series(0.0, index=test_df.index)) / 100.0),
        ("Supervised Classifier (Level 3)", (test_df.get("outbreak_probability", pd.Series(0.0, index=test_df.index)) >= 0.5).astype(int), test_df.get("outbreak_probability", pd.Series(0.0, index=test_df.index))),
    ]

    rows = []
    metrics_dict = {}

    for model_name, y_pred, y_prob in models_to_eval:
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5

        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            pr_auc = 0.0

        row = {
            "model": model_name,
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "false_positive_rate": round(float(fpr), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }
        rows.append(row)
        metrics_dict[model_name] = row

        logger.info(
            f"  {model_name} — Precision={p:.4f} | Recall={r:.4f} | "
            f"F1={f1:.4f} | ROC-AUC={auc:.4f} | FPR={fpr:.4f}"
        )

    comparison_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    comparison_df.to_csv(outbreak_reports_dir / "model_comparison.csv", index=False)
    comparison_df.to_csv(outbreak_reports_dir / "metrics.csv", index=False)

    with open(outbreak_reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    return comparison_df, metrics_dict


# ── 2. Outbreak Episode Grouping & Lead Time Calculation ─────────────────────

def group_outbreak_episodes(
    df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """Group continuous positive detections into outbreak episodes and calculate lead times.

    An alert is positive if `outbreak_status` in ["WATCH", "ELEVATED", "HIGH_ALERT"]
    or `overall_outbreak_score >= 25.0`.

    Max Gap Rule:
    -------------
    Allows gaps up to max_gap_days (7 days by default) between consecutive positive signals.

    Lead Time Calculation:
    ----------------------
    lead_time_days = peak_date - first_detection_date

    Returns
    -------
    episodes_df : DataFrame containing episode details and lead times
    """
    config = get_config(config_path)
    outbreak_reports_dir = config.outbreak_reports_dir
    max_gap_days = int(config.get("outbreak_detection.episode_grouping.max_gap_days", 7))

    logger.info(f"Grouping continuous outbreak episodes (max gap={max_gap_days}d)...")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "disease", "date"]).reset_index(drop=True)

    df["is_alert"] = df["outbreak_status"].isin(["WATCH", "ELEVATED", "HIGH_ALERT"])

    episodes = []
    episode_counter = 1

    for (region, disease), group in df.groupby(["region", "disease"]):
        group = group.reset_index(drop=True)

        current_episode: List[pd.Series] = []
        last_alert_date = None

        for idx, row in group.iterrows():
            if row["is_alert"]:
                if current_episode is None or len(current_episode) == 0:
                    current_episode = [row]
                else:
                    gap = (row["date"] - last_alert_date).days
                    if gap <= max_gap_days + 1:
                        current_episode.append(row)
                    else:
                        # Close current episode and start new
                        episodes.append(_build_episode_record(current_episode, episode_counter, region, disease))
                        episode_counter += 1
                        current_episode = [row]
                last_alert_date = row["date"]
            else:
                if current_episode and len(current_episode) > 0:
                    gap = (row["date"] - last_alert_date).days
                    if gap > max_gap_days:
                        episodes.append(_build_episode_record(current_episode, episode_counter, region, disease))
                        episode_counter += 1
                        current_episode = []

        if current_episode and len(current_episode) > 0:
            episodes.append(_build_episode_record(current_episode, episode_counter, region, disease))
            episode_counter += 1

    if not episodes:
        logger.warning("No outbreak episodes identified.")
        episodes_df = pd.DataFrame(columns=[
            "episode_id", "region", "disease", "start_date", "end_date", "duration_days",
            "peak_date", "peak_cases", "peak_growth_rate", "peak_incidence",
            "max_anomaly_score", "max_outbreak_probability", "lead_time_days"
        ])
    else:
        episodes_df = pd.DataFrame(episodes)

    episodes_df.to_csv(outbreak_reports_dir / "outbreak_lead_times.csv", index=False)
    logger.info(f"Identified {len(episodes_df)} outbreak episode(s). Lead times saved to {outbreak_reports_dir / 'outbreak_lead_times.csv'}")

    return episodes_df


def _build_episode_record(
    rows: List[pd.Series], counter: int, region: str, disease: str
) -> Dict[str, Any]:
    ep_df = pd.DataFrame(rows)
    start_date = ep_df["date"].min()
    end_date = ep_df["date"].max()
    duration = (end_date - start_date).days + 1

    peak_row = ep_df.loc[ep_df["new_cases"].idxmax()]
    peak_date = peak_row["date"]

    first_detection_date = start_date
    lead_time = (peak_date - first_detection_date).days

    return {
        "episode_id": f"EP_{region.upper()}_{disease.upper()}_{counter:03d}",
        "region": region,
        "disease": disease,
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "duration_days": int(duration),
        "peak_date": str(peak_date.date()),
        "peak_cases": int(peak_row["new_cases"]),
        "peak_growth_rate": round(float(peak_row.get("growth_rate_7d", 0.0)), 2),
        "peak_incidence": round(float(peak_row.get("incidence_per_100k", 0.0)), 2),
        "max_anomaly_score": round(float(ep_df["overall_outbreak_score"].max()), 2),
        "max_outbreak_probability": round(float(ep_df["outbreak_probability"].max()), 4),
        "lead_time_days": max(0, int(lead_time)),
    }


# ── 3. Visualisation Generator ───────────────────────────────────────────────

def generate_outbreak_visualizations(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> None:
    """Generate all required publication-quality Matplotlib plots."""
    config = get_config(config_path)
    fig_dir = config.figures_dir / "outbreak_detection"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating Phase 3 Visualisations in {fig_dir}...")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. Disease Cases + Baseline
    _plot_cases_and_baseline(df, fig_dir)

    # 2. Anomaly Score Timeline
    _plot_anomaly_timeline(df, fig_dir)

    # 3. Outbreak Probability Timeline
    _plot_probability_timeline(df, fig_dir)

    # 4. Outbreak Status Timeline
    _plot_status_timeline(df, fig_dir)

    # 5. ROC and PR Curves & Confusion Matrix (if test_df has targets)
    if "potential_outbreak" in test_df.columns:
        _plot_roc_and_pr_curves(test_df, fig_dir)
        _plot_confusion_matrix(test_df, fig_dir)

    # 6. Feature Importance Plot
    _plot_feature_importance_bar(config.outbreak_reports_dir, fig_dir)

    # 7. Lead Time Distribution
    _plot_lead_time_dist(episodes_df, fig_dir)

    logger.info("All Phase 3 visualisations completed.")


def _plot_cases_and_baseline(df: pd.DataFrame, out_dir: Path) -> None:
    diseases = df["disease"].unique()
    for disease in diseases:
        sub = df[df["disease"] == disease].sort_values("date")
        fig, ax = plt.subplots(figsize=(13, 6))

        ax.plot(sub["date"], sub["new_cases"], color="steelblue", lw=1.5, label="Daily Cases")
        if "baseline_mean" in sub.columns:
            ax.plot(sub["date"], sub["baseline_mean"], color="forestgreen", linestyle="--", lw=1.5, label="28-Day Baseline Mean")
        if "upper_threshold" in sub.columns:
            ax.plot(sub["date"], sub["upper_threshold"], color="darkorange", linestyle=":", lw=1.5, label="Upper Threshold (+2σ)")
        if "critical_threshold" in sub.columns:
            ax.plot(sub["date"], sub["critical_threshold"], color="crimson", linestyle=":", lw=1.5, label="Critical Threshold (+3σ)")

        # Highlight anomalies
        anomalies = sub[sub["statistical_anomaly"] != "normal"]
        ax.scatter(anomalies["date"], anomalies["new_cases"], color="crimson", s=30, zorder=5, label="Statistical Anomaly")

        ax.set_title(f"Early Outbreak Signal Baseline & Thresholds — {disease}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Daily New Cases", fontsize=11)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"cases_and_baseline_{disease.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()


def _plot_anomaly_timeline(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    diseases = df["disease"].unique()

    for disease in diseases:
        sub = df[df["disease"] == disease].sort_values("date")
        ax.plot(sub["date"], sub["overall_outbreak_score"], label=f"{disease} Score", lw=1.5)

    ax.axhline(50, color="darkorange", linestyle="--", label="Elevated Threshold (50)")
    ax.axhline(75, color="crimson", linestyle="--", label="High Alert Threshold (75)")
    ax.set_title("Overall Outbreak Anomaly Score Timeline (0–100)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Anomaly Score", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "anomaly_score_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_probability_timeline(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    diseases = df["disease"].unique()

    for disease in diseases:
        sub = df[df["disease"] == disease].sort_values("date")
        if "outbreak_probability" in sub.columns:
            ax.plot(sub["date"], sub["outbreak_probability"], label=f"{disease} Probability", lw=1.5)

    ax.axhline(0.5, color="crimson", linestyle="--", label="Decision Cutoff (0.5)")
    ax.set_title("Model-Estimated Outbreak Probability Timeline", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Outbreak Probability", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "outbreak_probability_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_status_timeline(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    status_map = {"NORMAL": 0, "WATCH": 1, "ELEVATED": 2, "HIGH_ALERT": 3}

    diseases = df["disease"].unique()
    for disease in diseases:
        sub = df[df["disease"] == disease].sort_values("date")
        y_vals = sub["outbreak_status"].map(status_map)
        ax.step(sub["date"], y_vals, where="post", label=disease, lw=1.5)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["NORMAL", "WATCH", "ELEVATED", "HIGH_ALERT"])
    ax.set_title("Outbreak Alert Status Timeline", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "outbreak_status_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_roc_and_pr_curves(test_df: pd.DataFrame, out_dir: Path) -> None:
    y_true = test_df["potential_outbreak"].values
    y_prob = test_df.get("outbreak_probability", pd.Series(0.0, index=test_df.index)).values

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC-AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("Supervised Outbreak Classifier — ROC Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="darkorange", lw=2, label=f"PR-AUC = {ap:.4f}")
    ax.axhline(y_true.mean(), color="grey", linestyle="--", label=f"Baseline = {y_true.mean():.3f}")
    ax.set_title("Supervised Outbreak Classifier — Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(test_df: pd.DataFrame, out_dir: Path) -> None:
    y_true = test_df["potential_outbreak"].values
    y_pred = (test_df.get("outbreak_probability", pd.Series(0.0, index=test_df.index)) >= 0.5).astype(int).values

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal (0)", "Outbreak (1)"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Outbreak Classifier — Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_feature_importance_bar(reports_dir: Path, out_dir: Path) -> None:
    fi_path = reports_dir / "feature_importance.csv"
    if not fi_path.exists():
        return
    fi_df = pd.read_csv(fi_path).head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="seagreen")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_lead_time_dist(episodes_df: pd.DataFrame, out_dir: Path) -> None:
    if episodes_df.empty or "lead_time_days" not in episodes_df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(episodes_df["lead_time_days"], bins=10, color="darkorange", edgecolor="white")
    ax.set_title("Early Warning Lead-Time Distribution (Days)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lead Time (Days from First Alert to Peak)")
    ax.set_ylabel("Number of Outbreak Episodes")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "lead_time_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 4. Comprehensive Markdown Report Writer ─────────────────────────────────

def write_evaluation_report(
    comparison_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> None:
    """Generate reports/outbreak_detection/model_evaluation_report.md."""
    config = get_config(config_path)
    out_path = config.outbreak_reports_dir / "model_evaluation_report.md"

    mean_lead = episodes_df["lead_time_days"].mean() if not episodes_df.empty else 0.0

    lines = [
        "# EpiWatch Phase 3 — Early Outbreak Detection Evaluation Report\n",
        "## 1. Executive Objective\n",
        "Build a multi-layered early-warning analytical pipeline to detect emerging epidemiological",
        "anomalies, score outbreak risk (0–100), and assign actionable alert statuses without claiming",
        "unconfirmed medical diagnoses.\n",
        "## 2. Architecture & Layering\n",
        "- **Level 1 (Baseline)**: Historical 28-day rolling mean + standard deviation thresholds (+2σ, +3σ).",
        "- **Level 2 (Statistical & Isolation Forest)**: Multi-signal normalization & unsupervised Isolation Forest.",
        "- **Level 3 (Supervised Classifier)**: Random Forest Classifier with Calibrated Probability Output.",
        "- **Signal Aggregator**: Weighted composite scoring (Statistical 25%, Isolation 20%, Supervised 35%, Growth 10%, Incidence 10%).\n",
        "## 3. Model Comparison Metrics\n",
    ]

    if not comparison_df.empty:
        lines.append("| Model | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate |")
        lines.append("|-------|-----------|--------|----------|---------|---------------------|")
        for _, row in comparison_df.iterrows():
            lines.append(
                f"| {row['model']} | {row['precision']} | {row['recall']} | {row['f1']} | "
                f"{row['roc_auc']} | {row['false_positive_rate']} |"
            )

    lines += [
        "\n## 4. Early Warning Lead Time Analysis\n",
        f"- **Identified Outbreak Episodes**: {len(episodes_df)}",
        f"- **Average Early Warning Lead Time**: {mean_lead:.1f} days before peak daily cases\n",
        "## 5. Important Project Limitations\n",
        "1. **Analytical Weak Labels**: `potential_outbreak` is a statistical heuristic label, NOT a medically confirmed outbreak ground truth.",
        "2. **Synthetic / Source-Calibrated Data**: Designed for capstone demonstration; real-world deployment requires official surveillance verification.",
        "3. **Statistical Heuristics**: System alert statuses (NORMAL, WATCH, ELEVATED, HIGH_ALERT) represent operational warning categories.",
        "4. **No Future Data Leakage**: All rolling features enforce strict `shift(1)` temporal availability.\n",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Evaluation report written to {out_path}")
