"""Tests for stress model."""
import pandas as pd
import numpy as np
from src.healthcare.stress_model import calculate_stress_score

def test_calculate_stress_score():
    df = pd.DataFrame({
        'bed_occupancy_ratio': [0.5, 0.9],
        'icu_occupancy_ratio': [0.4, 0.8],
        'available_oxygen_percent': [90, 20] # Note: high oxygen utilization = 10% vs 80%
    })
    
    weights = {
        'bed_occupancy': 30,
        'icu_occupancy': 30,
        'oxygen_utilization': 10
    }
    
    thresholds = {'moderate': 40, 'high': 60, 'critical': 80}
    
    res = calculate_stress_score(df, weights, thresholds)
    
    # Row 0:
    # bed = 50 * 30 = 1500
    # icu = 40 * 30 = 1200
    # oxy = (100 - 90 = 10) * 10 = 100
    # sum = 2800 / 70 = 40.0 -> MODERATE_STRESS
    
    assert np.isclose(res['healthcare_stress_score'].iloc[0], 40.0)
    assert res['stress_category'].iloc[0] == "MODERATE_STRESS"
    
    # Row 1:
    # bed = 90 * 30 = 2700
    # icu = 80 * 30 = 2400
    # oxy = (100 - 20 = 80) * 10 = 800
    # sum = 5900 / 70 = 84.28 -> CRITICAL_STRESS
    
    assert res['healthcare_stress_score'].iloc[1] > 80.0
    assert res['stress_category'].iloc[1] == "CRITICAL_STRESS"
