"""Environmental risk assessment."""
import pandas as pd
import numpy as np
from typing import Dict, Any

def assess_environmental(env_df: pd.DataFrame, region: str) -> Dict[str, Any]:
    """Extract Phase 6 environmental risk."""
    if env_df is None or env_df.empty:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing'}
        
    df = env_df[env_df['region'] == region]
    if df.empty:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing'}
        
    if 'date' in df.columns:
        df = df.sort_values('date')
        
    latest = df.iloc[-1]
    if 'environmental_risk_score' in latest and not pd.isna(latest['environmental_risk_score']):
        score = float(latest['environmental_risk_score'])
        # The env score is already 0-100
        score = min(max(score, 0.0), 100.0)
        return {
            'score': score,
            'raw': score,
            'status': 'available'
        }
        
    return {'score': np.nan, 'raw': np.nan, 'status': 'missing_indicators'}
