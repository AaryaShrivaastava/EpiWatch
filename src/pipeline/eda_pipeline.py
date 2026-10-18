"""EpiWatch Phase 2 EDA & Epidemiological Analytics Pipeline Orchestrator.

Pipeline Flow:
Processed/Feature Datasets -> Dataset Profiler -> Epidemiological Metrics ->
Time-Series & Growth Analysis -> Seasonality Analysis -> Environmental Correlations ->
Healthcare Burden Analysis -> Anomaly Detection & Outbreak Episodes ->
Visualizations -> Reports & Model Readiness Artifacts.
"""

from pathlib import Path
from typing import Dict
import pandas as pd

from src.analytics.anomaly_detection import detect_potential_anomalies, group_outbreak_episodes
from src.analytics.dataset_profiler import profile_datasets
from src.analytics.disease_analysis import analyze_growth_rates, analyze_time_series, analyze_yoy_trends
from src.analytics.environmental_analysis import analyze_environmental_correlations, calculate_correlation_matrices
from src.analytics.epidemiological_metrics import (
    calculate_cfr,
    calculate_disease_summary,
    calculate_hospitalizations,
    calculate_incidence,
    calculate_positivity
)
from src.analytics.healthcare_analysis import analyze_healthcare_burden
from src.analytics.seasonal_analysis import analyze_seasonality
from src.ingestion.loader import load_csv
from src.reports.analysis_report import generate_all_reports
from src.utils.config import get_config
from src.utils.logging_config import setup_logger
from src.visualization.environmental_plots import plot_environmental_visualizations
from src.visualization.epidemiological_plots import plot_epidemiological_visualizations
from src.visualization.healthcare_plots import plot_healthcare_visualizations

logger = setup_logger("eda_pipeline")


def run_eda_pipeline(config_path: str = "config.yaml") -> Dict[str, Path]:
    """Execute complete Phase 2 EDA and Epidemiological Analytics Pipeline."""
    logger.info("==================================================")
    logger.info("Starting EpiWatch Phase 2 EDA & Analytics Pipeline")
    logger.info("==================================================")

    config = get_config(config_path)

    # Output directories
    processed_dir = config.processed_data_dir
    features_dir = config.features_dir
    eda_dir = config.eda_dir
    figures_dir = config.figures_dir
    analytics_dir = config.analytics_data_dir

    for d in [eda_dir, figures_dir, analytics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load Clean Datasets & Feature Datasets
    logger.info("Loading Phase 1 cleaned and feature-engineered datasets...")
    df_epi_clean = load_csv(processed_dir / "epidemiological_clean.csv")
    df_env_clean = load_csv(processed_dir / "environmental_clean.csv")
    df_hc_clean = load_csv(processed_dir / "healthcare_clean.csv")
    df_demo_clean = load_csv(processed_dir / "demographics_clean.csv")
    master_df = load_csv(features_dir / "epiwatch_master_features.csv")

    # Population anchor
    bhopal_pop = 2368145.0
    if "population_2011" in df_demo_clean.columns:
        bhopal_pop = float(df_demo_clean["population_2011"].iloc[0])

    # 2. Dataset Profiling
    datasets_to_profile = {
        "epidemiological": df_epi_clean,
        "environmental": df_env_clean,
        "healthcare": df_hc_clean,
        "demographics": df_demo_clean,
        "master_features": master_df
    }
    profile_datasets(datasets_to_profile, eda_dir)

    # 3. Epidemiological Analytics
    disease_summary = calculate_disease_summary(df_epi_clean, analytics_dir)
    inc_df = calculate_incidence(df_epi_clean, bhopal_pop, analytics_dir)
    cfr_df = calculate_cfr(df_epi_clean, analytics_dir)
    hosp_df = calculate_hospitalizations(df_epi_clean, bhopal_pop, analytics_dir)
    pos_df = calculate_positivity(df_epi_clean, analytics_dir)

    # 4. Disease Time-Series, Growth & YoY Trends
    ts_moving_avg = analyze_time_series(df_epi_clean, analytics_dir)
    growth_df = analyze_growth_rates(df_epi_clean, analytics_dir, config_path=config_path)
    yoy_df = analyze_yoy_trends(df_epi_clean, analytics_dir)

    # 5. Seasonality Analysis
    m_season, w_season = analyze_seasonality(df_epi_clean, analytics_dir)

    # 6. Environmental Correlations & Heatmaps
    env_corr_df = analyze_environmental_correlations(master_df, analytics_dir)
    epi_matrix, env_matrix, comb_matrix = calculate_correlation_matrices(master_df, analytics_dir)

    # 7. Healthcare Burden Analysis
    hc_burden_df = analyze_healthcare_burden(master_df, analytics_dir)

    # 8. Statistical Anomalies & Outbreak Episodes
    anom_df = detect_potential_anomalies(df_epi_clean, analytics_dir, config_path=config_path)
    episodes_df = group_outbreak_episodes(anom_df, analytics_dir)

    # 9. Visualizations Generation
    plot_epidemiological_visualizations(
        df_epi=df_epi_clean,
        ts_moving_avg=ts_moving_avg,
        inc_df=inc_df,
        cfr_df=cfr_df,
        hosp_df=hosp_df,
        pos_df=pos_df,
        season_df=m_season,
        growth_df=growth_df,
        anom_df=anom_df,
        output_dir=figures_dir
    )

    plot_environmental_visualizations(
        env_corr_df=env_corr_df,
        epi_matrix=epi_matrix,
        env_matrix=env_matrix,
        comb_matrix=comb_matrix,
        output_dir=figures_dir
    )

    plot_healthcare_visualizations(
        hc_burden_df=hc_burden_df,
        output_dir=figures_dir
    )

    # 10. Reports Generation
    generate_all_reports(disease_summary, eda_dir)

    logger.info("==================================================")
    logger.info("EpiWatch Phase 2 EDA Pipeline Completed Successfully!")
    logger.info("==================================================")

    return {
        "eda_dir": eda_dir,
        "figures_dir": figures_dir,
        "analytics_dir": analytics_dir
    }


if __name__ == "__main__":
    run_eda_pipeline()
