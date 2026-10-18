"""Tests for data validation."""
import pandas as pd
from src.healthcare.data_validation import validate_healthcare_data

def test_validate_empty():
    df = pd.DataFrame()
    issues = validate_healthcare_data(df)
    assert len(issues) == 1
    assert "empty" in issues[0].lower()

def test_validate_negative():
    df = pd.DataFrame({
        'date': ['2024-01-01'],
        'region': ['Bhopal'],
        'hospital_beds': [-10]
    })
    issues = validate_healthcare_data(df)
    assert any("Negative" in issue for issue in issues)

def test_validate_duplicates():
    df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-01'],
        'region': ['Bhopal', 'Bhopal'],
        'hospital_beds': [100, 100]
    })
    issues = validate_healthcare_data(df)
    assert any("Duplicate" in issue for issue in issues)
