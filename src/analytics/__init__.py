"""Analytics package for EpiWatch Phase 2."""

from src.analytics.dataset_profiler import profile_datasets
from src.analytics.epidemiological_metrics import calculate_disease_summary, calculate_incidence, calculate_cfr, calculate_hospitalizations, calculate_positivity
from src.analytics.disease_analysis import analyze_time_series, analyze_growth_rates, analyze_yoy_trends
from src.analytics.seasonal_analysis import analyze_seasonality
from src.analytics.environmental_analysis import analyze_environmental_correlations, calculate_correlation_matrices
from src.analytics.healthcare_analysis import analyze_healthcare_burden
from src.analytics.anomaly_detection import detect_potential_anomalies, group_outbreak_episodes

__all__ = [
    "profile_datasets",
    "calculate_disease_summary",
    "calculate_incidence",
    "calculate_cfr",
    "calculate_hospitalizations",
    "calculate_positivity",
    "analyze_time_series",
    "analyze_growth_rates",
    "analyze_yoy_trends",
    "analyze_seasonality",
    "analyze_environmental_correlations",
    "calculate_correlation_matrices",
    "analyze_healthcare_burden",
    "detect_potential_anomalies",
    "group_outbreak_episodes"
]
