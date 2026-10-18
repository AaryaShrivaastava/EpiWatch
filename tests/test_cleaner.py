"""Unit tests for Data Cleaner and Missing Value Handler."""

import pandas as pd
from src.preprocessing.cleaner import DataCleaner


def test_snake_case_conversion():
    """Test column name standardization to snake_case."""
    assert DataCleaner.to_snake_case("NEW CASES") == "new_cases"
    assert DataCleaner.to_snake_case("New-Cases") == "new_cases"
    assert DataCleaner.to_snake_case("newCases") == "new_cases"
    assert DataCleaner.to_snake_case("   Air Quality Index (AQI)   ") == "air_quality_index_aqi"


def test_cleaner_pipeline():
    """Test data cleaner pipeline steps: snake_case, whitespace stripping, duplicates, sorting."""
    cleaner = DataCleaner()
    df_raw = pd.DataFrame({
        "Date": ["2025-01-02", "2025-01-01", "2025-01-01"],
        "Region ": [" Bhopal ", "Bhopal ", "Bhopal "],
        "New-Cases": [20, 10, 10]
    })

    df_clean, _ = cleaner.clean_dataset(
        df_raw,
        dataset_type="epidemiological",
        date_col="Date",
        group_cols=["Region"]
    )

    # Columns should be snake_case
    assert "date" in df_clean.columns
    assert "region" in df_clean.columns
    assert "new_cases" in df_clean.columns

    # Whitespace stripped
    assert list(df_clean["region"].unique()) == ["Bhopal"]

    # Duplicates removed (1 row dropped)
    assert len(df_clean) == 2

    # Chronologically sorted
    assert str(df_clean["date"].iloc[0].strftime("%Y-%m-%d")) == "2025-01-01"
    assert str(df_clean["date"].iloc[1].strftime("%Y-%m-%d")) == "2025-01-02"
