"""Tests for risk scoring."""
import pandas as pd
import numpy as np
from src.environmental_risk.risk_scoring import generate_risk_score

def test_generate_risk_score():
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3),
        'rainfall_mm': [0, 50, 100],
        'humidity_percent': [40, 60, 80],
        'temperature_c': [20, 30, 40],
        'rainfall_mm_14d_sum': [0, 50, 100],
        'humidity_percent_14d_mean': [40, 60, 80],
        'temperature_c_14d_mean': [20, 30, 40]
    })
    
    weights = {
        'rainfall_anomaly': 40,
        'humidity_anomaly': 30,
        'temperature_anomaly': 30
    }
    
    thresholds = {
        'moderate': 40,
        'high': 60,
        'very_high': 80
    }
    
    scored = generate_risk_score(df, weights, thresholds)
    
    assert 'environmental_risk_score' in scored.columns
    assert 'risk_category' in scored.columns
    
    # All scores should be between 0 and 100
    assert (scored['environmental_risk_score'] >= 0).all()
    assert (scored['environmental_risk_score'] <= 100).all()
    
def test_missing_indicator_handling():
    # Only rainfall is provided, but weights expect 3
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=3),
        'rainfall_mm': [0, 50, 100],
        'rainfall_mm_14d_sum': [0, 50, 100]
    })
    
    weights = {
        'rainfall_anomaly': 40,
        'humidity_anomaly': 30,
        'temperature_anomaly': 30
    }
    
    scored = generate_risk_score(df, weights, {})
    
    assert 'environmental_risk_score' in scored.columns
    # Should scale based only on the available weights
    assert not scored['environmental_risk_score'].isna().all()
    
    assert 'humidity_anomaly' in scored.iloc[0]['missing_indicators']
    assert 'rainfall_anomaly' in scored.iloc[0]['available_indicators']
