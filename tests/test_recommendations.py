"""Tests for recommendations module."""
import pandas as pd
from src.recommendations.evidence_builder import extract_evidence
from src.recommendations.rule_engine import evaluate_rules

def test_extract_evidence():
    outbreak = pd.DataFrame({'region': ['A'], 'date': ['2024-01-01'], 'anomaly_score': [2.5]})
    forecast = pd.DataFrame({'region': ['A'], 'forecast_growth_rate': [0.3]})
    gap = pd.DataFrame({'region': ['A'], 'resource_type': ['hospital_beds'], 'capacity_gap': [-50]})
    
    datasets = {
        'outbreak_path': outbreak,
        'forecast_path': forecast,
        'resource_gap_path': gap
    }
    
    ev = extract_evidence(datasets, 'A')
    
    assert ev['outbreak_anomaly_score'] == 2.5
    assert ev['forecast_growth_rate'] == 0.3
    assert ev['hospital_bed_gap'] == -50.0

def test_evaluate_rules():
    evidence = {
        'outbreak_anomaly_score': 3.0,
        'hospital_bed_gap': -100
    }
    
    thresholds = {
        'outbreak_monitoring': {'thresholds': {'anomaly_score_min': 2.0}}
    }
    
    recs = evaluate_rules(evidence, thresholds, 'A')
    
    assert len(recs) == 2
    rec_ids = [r['rec_id'] for r in recs]
    assert 'REC_OUTBREAK_001' in rec_ids
    assert 'REC_HC_GAP_001' in rec_ids
    
    gap_rec = [r for r in recs if r['rec_id'] == 'REC_HC_GAP_001'][0]
    assert gap_rec['priority'] == 'Critical review'
