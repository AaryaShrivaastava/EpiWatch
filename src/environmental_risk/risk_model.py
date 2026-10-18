"""Machine Learning Risk Model."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any, Tuple

def train_evaluate_ml_model(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    split_date: str,
    n_estimators: int = 100,
    max_depth: int = 10
) -> Dict[str, Any]:
    """Train an ML model with chronological split to prevent leakage."""
    
    available_preds = [p for p in predictors if p in df.columns]
    if not available_preds or target not in df.columns:
        return {'success': False, 'message': 'Missing data.'}
        
    df = df.sort_values('date').copy()
    
    train_mask = df['date'] < split_date
    test_mask = df['date'] >= split_date
    
    train_df = df[train_mask].dropna(subset=[target] + available_preds)
    test_df = df[test_mask].dropna(subset=[target] + available_preds)
    
    if len(train_df) < 30 or len(test_df) < 10:
        return {'success': False, 'message': 'Insufficient samples after split and NA drop.'}
        
    X_train = train_df[available_preds]
    y_train = train_df[target]
    
    X_test = test_df[available_preds]
    y_test = test_df[target]
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Handle negative predictions (clip at 0)
    predictions = [max(0, p) for p in predictions]
    
    # Calculate baseline (historical mean of training)
    baseline_pred = y_train.mean()
    baseline_preds = [baseline_pred] * len(y_test)
    
    return {
        'success': True,
        'model': model,
        'feature_importances': dict(zip(available_preds, model.feature_importances_)),
        'y_test': y_test.values,
        'predictions': predictions,
        'baseline_preds': baseline_preds,
        'test_dates': test_df['date'].values,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    }
