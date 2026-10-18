"""Forecast assessment."""
import pandas as pd
import numpy as np
from typing import Dict, Any

def assess_forecast_trajectory(forecast_df: pd.DataFrame, region: str) -> Dict[str, Any]:
    """Assess whether disease is expected to grow."""
    if forecast_df is None or forecast_df.empty:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing'}
        
    if 'region' in forecast_df.columns:
        df = forecast_df[forecast_df['region'] == region]
    else:
        df = forecast_df
        
    if df.empty:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing'}
        
    if 'date' in df.columns:
        df = df.sort_values('date')
        
    latest = df.iloc[-1]
    
    if 'forecast_growth_rate' in latest:
        raw = float(latest['forecast_growth_rate'])
        # Provisional mapping: growth > 0.5 (50%) => 100 risk
        score = (max(0, raw) / 0.5) * 100.0
        score = min(score, 100.0)
    elif 'forecast_surge_score' in latest:
        raw = float(latest['forecast_surge_score'])
        score = min(max(raw, 0.0), 100.0)
    else:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing_indicators'}
        
    return {
        'score': float(score),
        'raw': raw,
        'status': 'available'
    }
