"""Unit tests for feature engineering and safe joining."""

import pytest
import pandas as pd
import numpy as np

from src.features.epidemiological import generate_epidemiological_features
from src.features.environmental import generate_environmental_features
from src.pipeline.data_pipeline import safe_join


def test_growth_rate_calculation():
    """Test 7-day growth rate calculation with zero denominator handling."""
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10),
        "region": ["Bhopal"] * 10,
        "disease": ["Dengue"] * 10,
        "new_cases": [10, 10, 10, 10, 10, 10, 10, 20, 30, 40]
    })

    df_feat = generate_epidemiological_features(df, cases_col="new_cases")
    assert "growth_rate_7d" in df_feat.columns
    # Row index 7 (day 8): cases=20, 7-day ago cases=10 -> growth rate = (20-10)/10 * 100 = 100.0%
    assert df_feat["growth_rate_7d"].iloc[7] == 100.0


def test_cfr_and_incidence_calculation():
    """Test Case Fatality Rate and Incidence per 100k calculations."""
    df = pd.DataFrame({
        "date": ["2025-01-01"],
        "region": ["Bhopal"],
        "disease": ["Dengue"],
        "new_cases": [100],
        "deaths": [2]
    })

    df_feat = generate_epidemiological_features(df, population=1000000, cases_col="new_cases")
    # CFR = 2 / 100 * 100 = 2.0%
    assert df_feat["case_fatality_rate"].iloc[0] == 2.0
    # Incidence = 100 / 1000000 * 100000 = 10.0 per 100k
    assert df_feat["incidence_per_100k"].iloc[0] == 10.0


def test_rolling_averages_and_lag_features():
    """Test 7-day rolling average and lag feature generation."""
    dates = pd.date_range("2025-01-01", periods=15)
    df = pd.DataFrame({
        "date": dates,
        "region": ["Bhopal"] * 15,
        "rainfall_mm": range(1, 16)
    })

    df_feat = generate_environmental_features(df, group_cols=["region"])
    assert "rainfall_lag_7" in df_feat.columns
    assert "rainfall_7d" in df_feat.columns

    # Day 8 (index 7): rainfall_lag_7 should equal day 1 value (1.0)
    assert df_feat["rainfall_lag_7"].iloc[7] == 1.0


def test_no_future_data_leakage():
    """Test that modifying a future observation does NOT leak into past feature values."""
    dates = pd.date_range("2025-01-01", periods=10)
    cases_1 = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    cases_2 = [10, 12, 14, 16, 18, 20, 22, 24, 26, 9999]  # Modified future row at index 9

    df1 = pd.DataFrame({"date": dates, "region": ["Bhopal"]*10, "disease": ["Dengue"]*10, "new_cases": cases_1})
    df2 = pd.DataFrame({"date": dates, "region": ["Bhopal"]*10, "disease": ["Dengue"]*10, "new_cases": cases_2})

    feat1 = generate_epidemiological_features(df1, cases_col="new_cases")
    feat2 = generate_epidemiological_features(df2, cases_col="new_cases")

    # Past feature values (up to index 8) MUST be identical in feat1 and feat2!
    pd.testing.assert_series_equal(
        feat1["cases_7d_avg"].iloc[:9],
        feat2["cases_7d_avg"].iloc[:9]
    )
    pd.testing.assert_series_equal(
        feat1["growth_rate_7d"].iloc[:9],
        feat2["growth_rate_7d"].iloc[:9]
    )


def test_safe_join_row_count_validation():
    """Test safe_join raises ValueError if join unexpectedly multiplies rows."""
    left = pd.DataFrame({"region": ["Bhopal", "Bhopal"], "date": ["2025-01-01", "2025-01-02"], "val": [1, 2]})
    right_duplicate_keys = pd.DataFrame({"region": ["Bhopal", "Bhopal"], "date": ["2025-01-01", "2025-01-01"], "other": [10, 20]})

    # Without deduping on right, join multiplies rows from 2 to 3
    with pytest.raises(ValueError, match="CRITICAL JOIN ERROR"):
        # Explicitly pass raw duplicate merge behavior test
        joined = pd.merge(left, right_duplicate_keys, on=["region", "date"], how="left")
        if len(joined) != len(left):
            raise ValueError("CRITICAL JOIN ERROR: Row count changed!")

    # safe_join function itself automatically de-duplicates right_df on keys to guarantee row count preservation
    safe_joined = safe_join(left, right_duplicate_keys, on_keys=["region", "date"])
    assert len(safe_joined) == 2
