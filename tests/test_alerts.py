"""Tests for the Alerts Engine."""
import pandas as pd
from pathlib import Path
from src.alerts.rule_evaluator import evaluate_rules
from src.alerts.deduplication import apply_deduplication

def test_evaluate_rules():
    datasets = {
        'outbreak_path': pd.DataFrame({'region': ['A'], 'date': ['2024-01-01'], 'anomaly_score': [3.5]})
    }
    rules = {
        'outbreak_signal': {
            'enabled': True,
            'category': 'test',
            'severity': 'High',
            'threshold': 2.0
        }
    }
    
    candidates = evaluate_rules(datasets, 'A', rules)
    assert len(candidates) == 1
    assert candidates[0]['rule_id'] == 'outbreak_signal'

def test_apply_deduplication(tmp_path):
    from datetime import datetime
    candidates = [
        {'rule_id': 'outbreak_signal', 'region': 'A', 'timestamp': datetime.utcnow().isoformat()}
    ]
    rules = {
        'outbreak_signal': {'cooldown_hours': 24}
    }
    
    state_file = tmp_path / "alerts.csv"
    
    # Run 1: Should emit
    alerts1 = apply_deduplication(candidates, state_file, rules)
    assert len(alerts1) == 1
    
    # Save state
    pd.DataFrame(alerts1).to_csv(state_file, index=False)
    
    # Run 2: Should suppress because of cooldown
    alerts2 = apply_deduplication(candidates, state_file, rules)
    assert len(alerts2) == 0
