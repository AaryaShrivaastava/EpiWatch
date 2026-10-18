"""Tests for capacity metrics."""
import pandas as pd
import numpy as np
from src.healthcare.capacity_metrics import calculate_capacity_metrics

def test_calculate_capacity_metrics():
    df = pd.DataFrame({
        'hospital_beds': [3000],
        'icu_beds': [300],
        'doctors': [1000]
    })
    pop = 1000000.0 # 1 million
    
    res = calculate_capacity_metrics(df, pop)
    
    assert np.isclose(res['beds_per_100k'].iloc[0], 300.0)
    assert np.isclose(res['icu_beds_per_100k'].iloc[0], 30.0)
    assert np.isclose(res['doctors_per_100k'].iloc[0], 100.0)

def test_calculate_capacity_metrics_zero_pop():
    df = pd.DataFrame({'hospital_beds': [3000]})
    res = calculate_capacity_metrics(df, 0)
    assert pd.isna(res['beds_per_100k'].iloc[0])
