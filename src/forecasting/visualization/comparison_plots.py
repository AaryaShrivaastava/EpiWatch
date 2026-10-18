"""Model Error Comparison & Horizon Performance Plots."""

from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("comparison_plots")


def plot_model_comparison_bar(
    comp_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "model_comparison_bar.png",
) -> Path:
    """Bar chart comparing validation MAE across models and horizons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if comp_df.empty:
        return output_dir / filename

    fig, ax = plt.subplots(figsize=(10, 6))

    horizons = comp_df["horizon"].unique()
    models = comp_df["model"].unique()

    # Plot MAE bar per model
    sub = comp_df.groupby("model")["MAE"].mean().sort_values()
    ax.barh(sub.index[::-1], sub.values[::-1], color="seagreen", edgecolor="white")

    ax.set_title("Average Validation MAE by Forecasting Model", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Absolute Error (MAE)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    out_file = output_dir / filename
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    return out_file
