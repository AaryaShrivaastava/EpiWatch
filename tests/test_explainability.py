"""Tests for explainability module."""
import pandas as pd
import numpy as np
import pytest
from src.explainability.global_explanations import calculate_global_importance
from src.explainability.local_explanations import generate_local_explanation

class MockSHAPValues:
    def __init__(self, values):
        self.values = values

def test_calculate_global_importance():
    # 2 samples, 3 features
    vals = np.array([
        [1.0, -2.0, 0.5],
        [3.0, 4.0, -0.5]
    ])
    shap_vals = MockSHAPValues(vals)
    features = ['f1', 'f2', 'f3']
    
    df = calculate_global_importance(shap_vals, features)
    
    # mean abs:
    # f1: (1+3)/2 = 2.0
    # f2: (2+4)/2 = 3.0
    # f3: (0.5+0.5)/2 = 0.5
    
    assert len(df) == 3
    assert df.iloc[0]['feature'] == 'f2'
    assert df.iloc[0]['mean_abs_shap'] == 3.0

def test_generate_local_explanation():
    shap_vals = np.array([1.5, -0.5, 0.0])
    feat_vals = np.array([10, 20, 30])
    features = ['f1', 'f2', 'f3']
    
    exp = generate_local_explanation(
        identifier="test_1",
        shap_vals=shap_vals,
        feature_vals=feat_vals,
        feature_names=features,
        expected_value=0.5,
        actual_pred=1.5
    )
    
    assert exp['observation_id'] == "test_1"
    assert exp['top_contributions'][0]['feature'] == 'f1'
    assert exp['top_contributions'][0]['direction'] == 'up'
    
    assert exp['top_contributions'][1]['feature'] == 'f2'
    assert exp['top_contributions'][1]['direction'] == 'down'
