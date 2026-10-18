"""Publication-Grade Forecast Plots (Historical + Forecast + Prediction Interval)."""

from pathlib import Path
from typing import List, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("forecast_plots")


def plot_forecast_trajectory(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    region: str,
    disease: str,
    horizon: int,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Renders a publication-grade plot of historical cases, forecast line, and prediction interval."""
    output_dir.mkdir(parents=True, exist_ok=True)

    hist_sub = history_df[(history_df["region"] == region) & (history_df["disease"] == disease)].sort_values("date").tail(90)
    fc_sub = forecast_df[(forecast_df["region"] == region) & (forecast_df["disease"] == disease) & (forecast_df["horizon"] == horizon)].sort_values("date")

    if hist_sub.empty or fc_sub.empty:
        return output_dir

    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical line
    ax.plot(pd.to_datetime(hist_sub["date"]), hist_sub["new_cases"], color="steelblue", lw=2, label="Historical Cases")

    # Forecast line
    fc_dates = pd.to_datetime(fc_sub["date"])
    ax.plot(fc_dates, fc_sub["predicted_cases"], color="crimson", linestyle="--", lw=2.5, label=f"Forecast ({model_name})")

    # Prediction Interval (shaded region)
    ax.fill_between(
        fc_dates,
        fc_sub["lower_bound"],
        fc_sub["upper_bound"],
        color="crimson",
        alpha=0.20,
        label="95% Prediction Interval",
    )

    # Surge indicator annotation if active
    if bool(fc_sub["forecast_surge_signal"].iloc[0]):
        ax.annotate(
            "⚠️ Potential Surge Warning",
            xy=(fc_dates.iloc[0], fc_sub["predicted_cases"].iloc[0]),
            xytext=(fc_dates.iloc[0], fc_sub["predicted_cases"].max() * 1.1),
            arrowprops=dict(facecolor="orange", shrink=0.05, width=1, headwidth=6),
            fontsize=10,
            fontweight="bold",
            color="darkred",
        )

    title = f"{disease} Cases — {region} — {horizon}-Day Forecast ({model_name})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Daily New Cases", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_file = output_dir / f"forecast_{region}_{disease}_{horizon}d.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    return out_file
