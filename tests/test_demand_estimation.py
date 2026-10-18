"""Tests for demand estimation."""
import pandas as pd
from src.healthcare.demand_estimation import estimate_future_demand

def test_estimate_future_demand():
    forecast = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'predicted_cases': [1000] * 10
    })
    
    # hosp_rate 10%, icu_rate 2%
    res = estimate_future_demand(forecast, 0.1, 0.02, los_days=7, icu_los_days=10)
    
    # Daily hosp = 100, ICU = 20
    assert res['projected_hospital_admissions'].iloc[0] == 100.0
    assert res['projected_icu_admissions'].iloc[0] == 20.0
    
    # Day 7 occupied beds = 700
    assert res['expected_occupied_beds'].iloc[6] == 700.0
    # Day 10 occupied ICU = 200
    assert res['expected_occupied_icu_beds'].iloc[9] == 200.0
