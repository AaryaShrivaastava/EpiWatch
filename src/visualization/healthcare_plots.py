"""Healthcare Burden Visualization Module for EpiWatch Phase 2."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("healthcare_plots")


def plot_healthcare_visualizations(
    hc_burden_df: pd.DataFrame,
    output_dir: Path
):
    """Generate healthcare capacity and burden visualization.

    Figure generated in output_dir (reports/figures/):
    - healthcare_burden.png
    """
    logger.info("Generating healthcare burden visualization...")
    output_dir.mkdir(parents=True, exist_ok=True)

    if hc_burden_df.empty:
        logger.warning("Healthcare burden DataFrame is empty. Skipping plot.")
        return

    df = hc_burden_df.copy()
    if "month_str" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_str"])
        df.sort_values("month_date", inplace=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Plot 1: Hospital Beds & ICU Beds per 100k
    if "beds_per_100k" in df.columns:
        axes[0, 0].plot(df["month_date"], df["beds_per_100k"], label="Beds / 100k", color="tab:blue", linewidth=2)
    if "icu_beds_per_100k" in df.columns:
        axes[0, 0].plot(df["month_date"], df["icu_beds_per_100k"], label="ICU Beds / 100k", color="tab:red", linewidth=2)
    axes[0, 0].set_title("Hospital & ICU Beds per 100k Population", fontweight="bold")
    axes[0, 0].set_ylabel("Capacity per 100k")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Doctors per 100k
    if "doctors_per_100k" in df.columns:
        axes[0, 1].plot(df["month_date"], df["doctors_per_100k"], color="tab:green", linewidth=2)
    axes[0, 1].set_title("Doctors per 100k Population", fontweight="bold")
    axes[0, 1].set_ylabel("Doctors per 100k")
    axes[0, 1].grid(True)

    # Plot 3: Oxygen & Medicine Availability %
    if "available_oxygen_percent" in df.columns:
        axes[1, 0].plot(df["month_date"], df["available_oxygen_percent"], label="Oxygen Availability %", color="tab:cyan", linewidth=1.8)
    if "medicine_availability_percent" in df.columns:
        axes[1, 0].plot(df["month_date"], df["medicine_availability_percent"], label="Medicine Availability %", color="tab:purple", linewidth=1.8)
    axes[1, 0].set_title("Resource Availability (%)", fontweight="bold")
    axes[1, 0].set_ylabel("Percent (%)")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot 4: Hospitalizations & ICU Admissions vs Capacity
    if "hospitalizations" in df.columns:
        axes[1, 1].plot(df["month_date"], df["hospitalizations"], label="Hospitalizations", color="tab:orange", linewidth=2)
    if "icu_admissions" in df.columns:
        axes[1, 1].plot(df["month_date"], df["icu_admissions"], label="ICU Admissions", color="tab:red", linestyle="--", linewidth=2)
    axes[1, 1].set_title("Monthly Hospitalizations & ICU Pressure", fontweight="bold")
    axes[1, 1].set_ylabel("Patient Count")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle("EpiWatch — Healthcare System Stress & Capacity Analysis (Bhopal)", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(output_dir / "healthcare_burden.png", dpi=300)
    plt.close()
    logger.info("Saved: healthcare_burden.png")
