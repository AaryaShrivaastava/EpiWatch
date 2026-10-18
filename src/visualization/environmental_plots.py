"""Environmental Visualization Module for EpiWatch Phase 2."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("environmental_plots")


def plot_environmental_visualizations(
    env_corr_df: pd.DataFrame,
    epi_matrix: pd.DataFrame,
    env_matrix: pd.DataFrame,
    comb_matrix: pd.DataFrame,
    output_dir: Path
):
    """Generate environmental correlation plots and correlation heatmaps.

    Figures generated in output_dir (reports/figures/):
    - environmental_correlations.png
    - epidemiological_correlation_heatmap.png
    - environmental_correlation_heatmap.png
    - combined_correlation_heatmap.png
    """
    logger.info("Generating environmental visualizations and correlation heatmaps...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Environmental Lagged Correlations Bar Plot
    if not env_corr_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=env_corr_df,
            x="feature",
            y="pearson_correlation",
            hue="lag_days",
            palette="viridis"
        )
        plt.title("Pearson Correlation Between Disease Cases and Lagged Environmental Features", fontsize=14, fontweight="bold")
        plt.xlabel("Environmental Feature")
        plt.ylabel("Pearson Correlation Coefficient")
        plt.axhline(0, color="gray", linestyle="--")
        plt.legend(title="Lag Days")
        plt.savefig(output_dir / "environmental_correlations.png", dpi=300)
        plt.close()
        logger.info("Saved: environmental_correlations.png")

    # 2. Epidemiological Correlation Heatmap
    if not epi_matrix.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(epi_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True)
        plt.title("Epidemiological Variables Correlation Heatmap", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.savefig(output_dir / "epidemiological_correlation_heatmap.png", dpi=300)
        plt.close()
        logger.info("Saved: epidemiological_correlation_heatmap.png")

    # 3. Environmental Correlation Heatmap
    if not env_matrix.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(env_matrix, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, cbar=True)
        plt.title("Environmental Variables Correlation Heatmap", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.savefig(output_dir / "environmental_correlation_heatmap.png", dpi=300)
        plt.close()
        logger.info("Saved: environmental_correlation_heatmap.png")

    # 4. Combined Correlation Heatmap
    if not comb_matrix.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(comb_matrix, annot=True, fmt=".2f", cmap="Spectral", vmin=-1, vmax=1, cbar=True)
        plt.title("Combined Epidemiological & Environmental Correlation Heatmap", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.savefig(output_dir / "combined_correlation_heatmap.png", dpi=300)
        plt.close()
        logger.info("Saved: combined_correlation_heatmap.png")
