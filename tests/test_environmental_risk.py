"""Tests for environmental models and evaluation."""
import pandas as pd
import numpy as np
from src.environmental_risk.statistical_analysis import fit_poisson_model, fit_negative_binomial_model
from src.environmental_risk.risk_model import train_evaluate_ml_model
from src.environmental_risk.evaluation import evaluate_predictions

def test_fit_poisson():
    # Generate some count data
    np.random.seed(42)
    df = pd.DataFrame({
        'rainfall': np.random.uniform(0, 10, 100)
    })
    # target = exp(0.5 + 0.1 * rainfall)
    df['cases'] = np.random.poisson(np.exp(0.5 + 0.1 * df['rainfall']))
    
    res = fit_poisson_model(df, 'cases', ['rainfall'])
    assert res['success'] is True
    assert 'rainfall' in res['params']

def test_train_evaluate_ml_model():
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'rainfall': np.random.uniform(0, 10, 100),
        'temp': np.random.uniform(20, 35, 100),
        'cases': np.random.randint(0, 50, 100)
    })
    
    res = train_evaluate_ml_model(df, 'cases', ['rainfall', 'temp'], split_date='2024-03-01')
    assert res['success'] is True
    assert len(res['predictions']) == len(res['y_test'])
    assert res['test_samples'] > 0

def test_evaluate_predictions():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 30]
    
    metrics = evaluate_predictions(y_true, y_pred)
    
    # MAE = (2 + 2 + 0) / 3 = 4/3
    assert np.isclose(metrics['MAE'], 4.0/3.0)
    
    # RMSE = sqrt((4 + 4 + 0) / 3) = sqrt(8/3)
    assert np.isclose(metrics['RMSE'], np.sqrt(8.0/3.0))
