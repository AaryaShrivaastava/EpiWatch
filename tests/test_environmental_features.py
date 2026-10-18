"""Tests for environmental features and data alignment."""
import pandas as pd
import numpy as np
from src.environmental_risk.data_alignment import align_datasets
from src.environmental_risk.feature_builder import build_features

def test_align_datasets_weekly():
    env = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=14), # 2 weeks
        'rainfall_mm': [1.0] * 14,
        'temperature_c': [25.0] * 14
    })
    epi = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=14),
        'new_cases': [5] * 14
    })
    
    aligned = align_datasets(env, epi, frequency='W')
    
    # 2 weeks
    assert len(aligned) == 2
    
    # Weekly rainfall sum = 7.0
    assert np.allclose(aligned['rainfall_mm'], 7.0)
    
    # Weekly temp mean = 25.0
    assert np.allclose(aligned['temperature_c'], 25.0)
    
    # Weekly new cases sum = 35
    assert np.allclose(aligned['new_cases'], 35.0)

def test_build_features():
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'rainfall_mm': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    
    variables = {
        'rainfall_mm': ['lag', 'mean', 'sum']
    }
    
    featured = build_features(df, variables, lags=[1], rolling_windows=[3])
    
    # Lag 1 for row 1 should be 1, row 2 should be 2
    assert pd.isna(featured.loc[0, 'rainfall_mm_lag_1'])
    assert featured.loc[1, 'rainfall_mm_lag_1'] == 1.0
    
    # 3d sum for row 2 should be 1+2+3 = 6
    assert featured.loc[2, 'rainfall_mm_3d_sum'] == 6.0
    
    # 3d mean for row 2 should be 2.0
    assert featured.loc[2, 'rainfall_mm_3d_mean'] == 2.0
    
def test_future_leakage_prevention():
    """Verify that changing future data does not change past features (No leakage)."""
    df_base = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'rainfall_mm': [10, 20, 30, 40, 50]
    })
    
    df_future = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'rainfall_mm': [10, 20, 30, 999, 999] # Future modified
    })
    
    variables = {'rainfall_mm': ['mean']}
    
    feat_base = build_features(df_base, variables, lags=[], rolling_windows=[3])
    feat_future = build_features(df_future, variables, lags=[], rolling_windows=[3])
    
    # The 3-day mean on day 3 (index 2) should depend only on days 1, 2, 3
    # and should be identical in both.
    assert feat_base.loc[2, 'rainfall_mm_3d_mean'] == feat_future.loc[2, 'rainfall_mm_3d_mean']
    assert feat_base.loc[2, 'rainfall_mm_3d_mean'] == 20.0
