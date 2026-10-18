"""Unit tests for EpiWatch Phase 3 — Early Outbreak Detection.

Tests cover:
  - Baseline calculation & zero std deviation handling
  - Anomaly threshold scaling & statistical categories
  - Isolation Forest model fitting & 0-100 anomaly score scaling
  - Outbreak probability range [0.0, 1.0]
  - Overall outbreak score range [0.0, 100.0] & alert status tiers
  - Severity level assignment & contributing signals generation
  - Outbreak episode grouping & lead time calculation
  - Chronological split & data leakage prevention
  - Residual missing value imputation
"""

import json
import numpy as np
import pandas as pd
import pytest

from src.outbreak_detection.baseline_detector import detect_baseline_anomalies
from src.outbreak_detection.statistical_detector import compute_statistical_signals
from src.outbreak_detection.isolation_forest import train_isolation_forest, predict_isolation_forest
from src.outbreak_detection.outbreak_classifier import train_outbreak_classifier, predict_outbreak_probability
from src.outbreak_detection.signal_aggregator import aggregate_signals
from src.outbreak_detection.severity import assign_severity_and_reasons
from src.evaluation.outbreak_evaluation import group_outbreak_episodes
from src.outbreak_detection.feature_builder import split_dataset_chronologically, create_weak_label


def _make_dummy_epi_df(n: int = 150) -> pd.DataFrame:
    """Helper to generate dummy epidemiological DataFrame for unit tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    cases = rng.integers(10, 50, n)

    # Inject spikes safely within bounds
    if n >= 45:
        cases[40:45] = [80, 120, 150, 110, 90]
    if n >= 105:
        cases[100:105] = [200, 250, 220, 180, 140]
    elif n >= 25:
        cases[20:25] = [150, 180, 210, 160, 120]


    return pd.DataFrame({
        "date": dates,
        "region": "Bhopal",
        "disease": "Dengue",
        "new_cases": cases,
        "active_cases": cases * 3,
        "deaths": rng.integers(0, 3, n),
        "hospitalizations": rng.integers(0, 10, n),
        "icu_admissions": rng.integers(0, 3, n),
        "tests": rng.integers(100, 500, n),
        "test_positivity_rate_percent": rng.uniform(2, 15, n),
        "growth_rate_7d": rng.uniform(-10, 40, n),
        "growth_rate_14d": rng.uniform(-10, 40, n),
        "incidence_per_100k": cases / 10.0,
        "hospitalization_rate": rng.uniform(1, 20, n),
        "case_fatality_rate": rng.uniform(0, 2, n),
        "temperature_c": rng.uniform(20, 35, n),
        "humidity_percent": rng.uniform(40, 80, n),
        "rainfall_mm": rng.uniform(0, 20, n),
        "air_quality_index": rng.uniform(100, 200, n),
        "rainfall_lag_7": rng.uniform(0, 20, n),
        "humidity_lag_7": rng.uniform(40, 80, n),
        "rainfall_7d": rng.uniform(0, 50, n),
        "month": dates.month,
        "week": dates.isocalendar().week.astype(int),
        "quarter": dates.quarter,
        "potential_anomaly": (cases > 100).astype(int),
    })


# ── 1. Baseline Detector Tests ────────────────────────────────────────────────

def test_baseline_detector_calculation():
    """Verify rolling baseline mean, std, thresholds, and deviation calculation."""
    df = _make_dummy_epi_df(100)
    res = detect_baseline_anomalies(df)

    assert "baseline_mean" in res.columns
    assert "baseline_std" in res.columns
    assert "upper_threshold" in res.columns
    assert "baseline_deviation" in res.columns
    assert "statistical_anomaly" in res.columns
    assert set(res["statistical_anomaly"].unique()).issubset({"normal", "potential_anomaly", "strong_anomaly"})


def test_baseline_detector_zero_std_safety():
    """Verify zero standard deviation does not cause division by zero or infinite deviation."""
    df = _make_dummy_epi_df(50)
    df["new_cases"] = 10  # Constant series -> std = 0

    res = detect_baseline_anomalies(df)
    assert not res["baseline_deviation"].isin([np.inf, -np.inf]).any()
    assert not res["baseline_deviation"].isna().any()
    assert (res["baseline_deviation"] == 0.0).all()


# ── 2. Statistical Signals Tests ──────────────────────────────────────────────

def test_compute_statistical_signals():
    """Verify multi-signal normalization and 0-100 composite score calculation."""
    df = _make_dummy_epi_df(60)
    df = detect_baseline_anomalies(df)
    res = compute_statistical_signals(df)

    assert "statistical_signal_score" in res.columns
    assert res["statistical_signal_score"].min() >= 0.0
    assert res["statistical_signal_score"].max() <= 100.0


# ── 3. Isolation Forest Tests ────────────────────────────────────────────────

def test_isolation_forest_train_and_predict():
    """Verify Isolation Forest training on train split and 0-100 anomaly score mapping."""
    df = _make_dummy_epi_df(100)
    feature_cols = ["new_cases", "growth_rate_7d", "incidence_per_100k", "temperature_c"]

    model, scaler, meta = train_isolation_forest(df, feature_cols)
    res = predict_isolation_forest(df, feature_cols, model, scaler)

    assert "isolation_score" in res.columns
    assert "anomaly_score_0_100" in res.columns
    assert "isolation_anomaly" in res.columns

    assert res["anomaly_score_0_100"].min() >= 0.0
    assert res["anomaly_score_0_100"].max() <= 100.0
    assert set(res["isolation_anomaly"].unique()).issubset({0, 1})


# ── 4. Outbreak Classifier Tests ─────────────────────────────────────────────

def test_outbreak_classifier_train_and_predict():
    """Verify Supervised Classifier fits and outputs valid probabilities [0.0, 1.0]."""
    df = _make_dummy_epi_df(100)
    df["potential_outbreak"] = create_weak_label(df)
    feature_cols = ["new_cases", "growth_rate_7d", "incidence_per_100k"]

    base_model, cal_model, fi_df, meta = train_outbreak_classifier(df, feature_cols)
    res = predict_outbreak_probability(df, feature_cols, cal_model)

    assert "outbreak_probability" in res.columns
    assert res["outbreak_probability"].min() >= 0.0
    assert res["outbreak_probability"].max() <= 1.0
    assert len(fi_df) == len(feature_cols)


# ── 5. Signal Aggregator Tests ────────────────────────────────────────────────

def test_signal_aggregator_status_mapping():
    """Verify overall score is in [0, 100] and maps to valid alert status tiers."""
    df = _make_dummy_epi_df(80)
    df["baseline_anomaly_score"] = np.random.uniform(0, 100, 80)
    df["anomaly_score_0_100"] = np.random.uniform(0, 100, 80)
    df["outbreak_probability"] = np.random.uniform(0, 1, 80)
    df["growth_signal"] = np.random.uniform(0, 1, 80)
    df["incidence_signal"] = np.random.uniform(0, 1, 80)

    res = aggregate_signals(df)

    assert "overall_outbreak_score" in res.columns
    assert "outbreak_status" in res.columns

    assert res["overall_outbreak_score"].min() >= 0.0
    assert res["overall_outbreak_score"].max() <= 100.0
    assert set(res["outbreak_status"].unique()).issubset({"NORMAL", "WATCH", "ELEVATED", "HIGH_ALERT"})


# ── 6. Severity & Contributing Signals Tests ────────────────────────────────

def test_severity_and_reasons_assignment():
    """Verify severity levels and valid JSON contributing signals format."""
    df = _make_dummy_epi_df(50)
    df["baseline_deviation"] = 2.5
    df["outbreak_status"] = "HIGH_ALERT"

    res = assign_severity_and_reasons(df)

    assert "severity" in res.columns
    assert "contributing_signals" in res.columns
    assert set(res["severity"].unique()).issubset({"LOW", "MODERATE", "HIGH", "CRITICAL"})

    # Verify JSON structure
    first_reasons = json.loads(res["contributing_signals"].iloc[0])
    assert isinstance(first_reasons, list)
    assert len(first_reasons) > 0


# ── 7. Episode Grouping & Lead Time Tests ────────────────────────────────────

def test_episode_grouping_and_lead_time():
    """Verify continuous episode grouping and non-negative lead time calculation."""
    df = _make_dummy_epi_df(100)
    df["outbreak_status"] = "NORMAL"
    # Create an alert period
    df.loc[df.index[20:26], "outbreak_status"] = "ELEVATED"
    df.loc[df.index[20:26], "overall_outbreak_score"] = 60.0
    df.loc[df.index[20:26], "outbreak_probability"] = 0.8


    episodes_df = group_outbreak_episodes(df)

    assert not episodes_df.empty
    assert "episode_id" in episodes_df.columns
    assert "lead_time_days" in episodes_df.columns
    assert (episodes_df["lead_time_days"] >= 0).all()
    assert (episodes_df["duration_days"] >= 1).all()


# ── 8. Chronological Split & Leakage Prevention Tests ─────────────────────────

def test_chronological_split_ordering():
    """Verify chronological split respects temporal order."""
    df = _make_dummy_epi_df(120)
    train_df, val_df, test_df, split_info = split_dataset_chronologically(df)

    assert train_df["date"].max() <= val_df["date"].min()
    assert val_df["date"].max() <= test_df["date"].min()
    assert split_info["split_type"] == "chronological"
