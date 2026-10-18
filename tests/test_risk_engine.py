"""Tests for risk engine."""
import pandas as pd
import numpy as np
from src.risk_engine.disease_burden import assess_disease_burden
from src.risk_engine.scoring import calculate_integrated_score, classify_risk
from src.risk_engine.risk_explanation import generate_risk_explanation

def test_assess_disease_burden():
    epi = pd.DataFrame({'region': ['A'], 'date': ['2024-01-01'], 'new_cases': [500]})
    demo = pd.DataFrame({'region': ['A'], 'population_2011': [1000000]})
    
    res = assess_disease_burden(epi, demo, 'A')
    
    assert res['status'] == 'available'
    assert res['raw'] == 50.0  # 500 / 1M * 100k
    assert res['score'] == 100.0  # mapped 50 -> 100

def test_assess_disease_burden_missing():
    res = assess_disease_burden(None, None, 'A')
    assert res['status'] == 'missing'
    assert pd.isna(res['score'])

def test_calculate_integrated_score():
    assessments = {
        'disease_burden': {'score': 100.0, 'status': 'available'},
        'environmental_risk': {'score': 50.0, 'status': 'available'},
        'outbreak_signal': {'score': np.nan, 'status': 'missing'}
    }
    
    weights = {
        'disease_burden': 50,
        'environmental_risk': 50,
        'outbreak_signal': 50
    }
    
    score, contribs = calculate_integrated_score(assessments, weights)
    
    # Missing is ignored, remaining weight = 100
    # Final = (100 * 50 + 50 * 50) / 100 = 75
    assert np.isclose(score, 75.0)
    assert np.isclose(contribs['disease_burden'], 50.0)
    assert np.isclose(contribs['environmental_risk'], 25.0)

def test_classify_risk():
    thresholds = {'moderate': 25, 'high': 50, 'very_high': 75}
    assert classify_risk(10, thresholds) == 'Low'
    assert classify_risk(30, thresholds) == 'Moderate'
    assert classify_risk(60, thresholds) == 'High'
    assert classify_risk(90, thresholds) == 'Very High'
    assert classify_risk(np.nan, thresholds) == 'Insufficient Data'

def test_risk_explanation():
    contribs = {'disease_burden': 50.0, 'healthcare_stress': 20.0}
    assessments = {'forecast_trajectory': {'status': 'available', 'score': 60}}
    text = generate_risk_explanation("A", "High", 70.0, assessments, contribs)
    
    assert "High" in text
    assert "disease burden" in text
    assert "healthcare stress" in text
    assert "forecast suggests further growth" in text
