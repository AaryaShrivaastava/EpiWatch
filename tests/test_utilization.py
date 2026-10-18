"""Tests for utilization metrics."""
import pandas as pd
import numpy as np
from src.healthcare.utilization import estimate_occupied_beds, calculate_utilization

def test_estimate_occupied_beds():
    epi = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'hospitalizations': [10] * 10,
        'icu_admissions': [2] * 10
    })
    hc = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'hospital_beds': [100] * 10
    })
    
    res = estimate_occupied_beds(epi, hc, los_days=7, icu_los=10)
    
    # On day 7 (index 6), it should be 70
    assert res.loc[6, 'estimated_occupied_beds'] == 70.0
    # On day 10 (index 9), it should be 70 because it's a 7-day rolling sum
    assert res.loc[9, 'estimated_occupied_beds'] == 70.0
    # ICU on day 10 should be 20
    assert res.loc[9, 'estimated_occupied_icu_beds'] == 20.0

def test_calculate_utilization():
    df = pd.DataFrame({
        'hospital_beds': [100],
        'estimated_occupied_beds': [75],
        'tests_performed': [500],
        'testing_capacity_tests_per_day': [1000]
    })
    
    res = calculate_utilization(df)
    
    assert res['bed_occupancy_ratio'].iloc[0] == 0.75
    assert res['bed_occupancy_percent'].iloc[0] == 75.0
    assert res['testing_utilization_ratio'].iloc[0] == 0.5
