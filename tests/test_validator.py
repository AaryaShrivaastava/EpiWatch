"""Unit tests for Data Validator."""

import pandas as pd
from src.ingestion.validator import DataValidator


def test_required_column_validation():
    """Test schema validation fails when a required column is missing."""
    schema_config = {
        "epidemiological": {
            "required_columns": ["date", "region", "disease", "new_cases"]
        }
    }
    validator = DataValidator(schema_config)

    # Missing 'disease' column
    df = pd.DataFrame({
        "date": ["2025-01-01"],
        "region": ["Bhopal"],
        "new_cases": [5]
    })

    passed, report = validator.validate_dataset(df, "epidemiological")
    assert not passed
    assert report["validation_status"] == "FAILED"
    assert len(report["schema_issues"]) > 0


def test_negative_values_detection():
    """Test validator flags negative case values."""
    schema_config = {
        "epidemiological": {
            "required_columns": ["date", "region", "new_cases"],
            "non_negative_columns": ["new_cases"]
        }
    }
    validator = DataValidator(schema_config)

    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "region": ["Bhopal", "Bhopal"],
        "new_cases": [10, -5]
    })

    passed, report = validator.validate_dataset(df, "epidemiological")
    assert report["validation_status"] in ["PASSED_WITH_WARNINGS", "FAILED"]
    assert report["invalid_rows_count"] >= 1
    assert any("negative" in issue for issue in report["invalid_value_issues"])


def test_duplicate_detection():
    """Test detection of duplicate records based on key columns."""
    schema_config = {
        "epidemiological": {
            "required_columns": ["date", "region", "disease"],
            "key_columns": ["date", "region", "disease"]
        }
    }
    validator = DataValidator(schema_config)

    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-01"],
        "region": ["Bhopal", "Bhopal"],
        "disease": ["Dengue", "Dengue"]
    })

    _, report = validator.validate_dataset(df, "epidemiological")
    assert report["duplicate_rows"] == 2


def test_humidity_range_check():
    """Test out-of-bounds humidity percentage detection."""
    schema_config = {
        "environmental": {
            "required_columns": ["date", "humidity_percent"],
            "range_checks": {"humidity_percent": [0.0, 100.0]}
        }
    }
    validator = DataValidator(schema_config)

    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "humidity_percent": [50.0, 150.0]
    })

    _, report = validator.validate_dataset(df, "environmental")
    assert report["invalid_rows_count"] == 1
    assert any("outside expected range" in issue for issue in report["invalid_value_issues"])
