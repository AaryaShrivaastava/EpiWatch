"""Residual Diagnostic Plots (Residual Distribution & Timeline)."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("residual_plots")


def plot_residuals_diagnostics(
    residuals: np.ndarray,
    model_name: str,
    output_dir: Path,
    filename: str = "residual_diagnostics.png",
) -> Path:
    """Renders histogram and scatter plot of forecast residuals."""
    output_dir.mkdir(parents=True, exist_ok=True)
    res = np.array(residuals, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Residual Histogram
    axes[0].hist(res, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", linestyle="--", lw=1)
    axes[0].set_title(f"Residual Distribution — {model_name}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    # 2. Residual Sequence Plot
    axes[1].plot(res, color="steelblue", lw=1.2, marker="o", ms=3)
    axes[1].axhline(0, color="red", linestyle="--", lw=1)
    axes[1].set_title("Residual Sequence Plot", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Validation Index")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_file = output_dir / filename
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    return out_file
