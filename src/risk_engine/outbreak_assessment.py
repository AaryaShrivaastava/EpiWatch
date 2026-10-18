"""Outbreak assessment."""
import pandas as pd
import numpy as np
from typing import Dict, Any

def assess_outbreak_signal(outbreak_df: pd.DataFrame, region: str) -> Dict[str, Any]:
    """Extract and normalize Phase 3 anomaly scores."""
    if outbreak_df is None or outbreak_df.empty:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing'}
        
    # Check if region is in df
    if 'region' in outbreak_df.columns:
        df = outbreak_df[outbreak_df['region'] == region]
    else:
        df = outbreak_df
        
    if df.empty:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing'}
        
    # Sort by date if possible
    if 'date' in df.columns:
        df = df.sort_values('date')
        
    latest = df.iloc[-1]
    
    # We look for anomaly_score or is_anomaly
    if 'anomaly_score' in latest:
        raw = float(latest['anomaly_score'])
        # If anomaly_score is Z-score like, mapping > 3 to 100
        score = (max(0, raw) / 3.0) * 100.0
        score = min(score, 100.0)
    elif 'is_anomaly' in latest:
        raw = float(latest['is_anomaly'])
        score = 100.0 if raw > 0 else 0.0
    else:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing_indicators'}
        
    return {
        'score': float(score),
        'raw': raw,
        'status': 'available'
    }
