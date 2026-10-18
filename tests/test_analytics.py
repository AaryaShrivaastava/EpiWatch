"""Unit tests for Phase 2 analytics and metrics functions."""

import numpy as np
import pandas as pd
from pathlib import Path

from src.analytics.epidemiological_metrics import (
    calculate_disease_summary,
    calculate_incidence,
    calculate_cfr,
    calculate_hospitalizations,
    calculate_positivity
)
from src.analytics.disease_analysis import (
    analyze_time_series,
    analyze_growth_rates,
    analyze_yoy_trends,
    classify_trend
)
from src.analytics.seasonal_analysis import analyze_seasonality
from src.analytics.environmental_analysis import analyze_environmental_correlations
from src.analytics.anomaly_detection import detect_potential_anomalies, group_outbreak_episodes


def test_calculate_incidence(tmp_path: Path):
    """Test incidence per 100k calculation across periods."""
    df_epi = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=14),
        "region": ["Bhopal"] * 14,
        "disease": ["Dengue"] * 14,
        "new_cases": [10] * 14
    })
    pop = 1000000.0  # 1M

    inc_df = calculate_incidence(df_epi, pop, tmp_path)
    assert not inc_df.empty
    assert "incidence_per_100k" in inc_df.columns
    # 10 cases / 1,000,000 * 100,000 = 1.0 incidence per 100k for weekly/daily sum
    w_inc = inc_df[inc_df["period"] == "weekly"]["incidence_per_100k"].iloc[0]
    assert w_inc > 0


def test_calculate_cfr_denominator_zero(tmp_path: Path):
    """Test Case Fatality Rate returns NaN when cases denominator is 0."""
    df_epi = pd.DataFrame({
        "date": ["2025-01-01"],
        "region": ["Bhopal"],
        "disease": ["Dengue"],
        "new_cases": [0],
        "deaths": [0]
    })

    cfr_df = calculate_cfr(df_epi, tmp_path)
    # CFR should be NaN when cases = 0
    val = cfr_df["cfr_percent"].iloc[0]
    assert pd.isna(val) or np.isnan(val)


def test_hospitalization_rate_calculation(tmp_path: Path):
    """Test hospitalization rate % calculation."""
    df_epi = pd.DataFrame({
        "date": ["2025-01-01"],
        "region": ["Bhopal"],
        "disease": ["Dengue"],
        "new_cases": [100],
        "hospitalizations": [25],
        "deaths": [1]
    })

    hosp_df = calculate_hospitalizations(df_epi, 1000000.0, tmp_path)
    assert hosp_df["hospitalization_rate_percent"].iloc[0] == 25.0


def test_trend_classification():
    """Test growth rate trend classification with config thresholds."""
    thresholds = {"rapid_increase": 30.0, "increase": 10.0, "decrease": -10.0, "rapid_decrease": -30.0}
    assert classify_trend(45.0, thresholds) == "Rapidly Increasing"
    assert classify_trend(15.0, thresholds) == "Increasing"
    assert classify_trend(0.0, thresholds) == "Stable"
    assert classify_trend(-15.0, thresholds) == "Decreasing"
    assert classify_trend(-50.0, thresholds) == "Rapidly Decreasing"


def test_yoy_trend_calculation(tmp_path: Path):
    """Test Year-over-Year (YoY %) trend calculation."""
    df_epi = pd.DataFrame({
        "date": ["2024-06-01", "2025-06-01"],
        "region": ["Bhopal", "Bhopal"],
        "disease": ["Dengue", "Dengue"],
        "new_cases": [100, 150]
    })

    yoy_df = analyze_yoy_trends(df_epi, tmp_path)
    assert len(yoy_df) == 2
    # YoY for 2025 = (150-100)/100 * 100 = 50.0%
    yoy_2025 = yoy_df[yoy_df["year"] == 2025]["yoy_change_percent"].iloc[0]
    assert yoy_2025 == 50.0


def test_seasonal_aggregation(tmp_path: Path):
    """Test monthly and weekly seasonal indices aggregation."""
    dates = pd.date_range("2025-01-01", periods=60)
    df_epi = pd.DataFrame({
        "date": dates,
        "region": ["Bhopal"] * 60,
        "disease": ["Dengue"] * 60,
        "new_cases": range(1, 61)
    })

    m_df, w_df = analyze_seasonality(df_epi, tmp_path)
    assert "seasonal_index" in m_df.columns
    assert "seasonal_index" in w_df.columns
    assert len(m_df) > 0


def test_anomaly_detection_and_episodes(tmp_path: Path):
    """Test potential statistical anomaly detection and episode grouping."""
    dates = pd.date_range("2025-01-01", periods=40)
    cases = [10] * 35 + [100, 120, 150, 10, 10]  # Anomaly surge on days 36, 37, 38

    df_epi = pd.DataFrame({
        "date": dates,
        "region": ["Bhopal"] * 40,
        "disease": ["Dengue"] * 40,
        "new_cases": cases,
        "growth_rate_7d": [0.0] * 40
    })

    anom_df = detect_potential_anomalies(df_epi, tmp_path)
    assert "potential_anomaly" in anom_df.columns
    assert anom_df["potential_anomaly"].sum() >= 1

    episodes_df = group_outbreak_episodes(anom_df, tmp_path)
    assert not episodes_df.empty
    assert "episode_id" in episodes_df.columns
    assert episodes_df["duration_days"].iloc[0] >= 1
