"""Tests for lag analysis."""
import pandas as pd
import numpy as np
from src.environmental_risk.lag_analysis import compute_lag_correlations

def test_compute_lag_correlations():
    # Make a synthetic dataset with perfect lag=1 correlation
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=20),
        'rainfall_mm': np.arange(20),
        'new_cases': [0] + list(np.arange(19)) # new_cases is rainfall shifted by 1
    })
    
    res = compute_lag_correlations(
        df, target='new_cases', variables=['rainfall_mm'], lags=[0, 1, 2],
        region='Test', disease='Test'
    )
    
    assert len(res) == 3
    
    # Lag 1 should have correlation near 1.0
    lag1_corr = res[res['lag'] == 1]['pearson_correlation'].values[0]
    assert np.isclose(lag1_corr, 1.0)
    
    # Lag 0 should be slightly less than 1.0
    lag0_corr = res[res['lag'] == 0]['pearson_correlation'].values[0]
    assert lag0_corr < 1.0
