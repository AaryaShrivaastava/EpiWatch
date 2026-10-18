"""Disease burden assessment."""
import pandas as pd
import numpy as np
from typing import Dict, Any

def assess_disease_burden(epi_df: pd.DataFrame, demo_df: pd.DataFrame, region: str) -> Dict[str, Any]:
    """Calculate population-adjusted disease burden score."""
    if epi_df is None or epi_df.empty or demo_df is None or demo_df.empty:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing'}
        
    epi = epi_df[epi_df['region'] == region].sort_values('date')
    demo = demo_df[demo_df['region'] == region]
    
    if epi.empty or demo.empty:
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing'}
        
    pop = float(demo.iloc[0]['population_2011'])
    if pop <= 0:
        return {'score': np.nan, 'raw': np.nan, 'status': 'invalid_population'}
        
    latest_cases = epi.iloc[-1].get('new_cases', np.nan)
    if pd.isna(latest_cases):
        return {'score': np.nan, 'raw': np.nan, 'status': 'missing_cases'}
        
    incidence = (latest_cases / pop) * 100000
    
    # Simple provisional thresholding: 
    # incidence > 50 -> 100
    score = (incidence / 50.0) * 100.0
    score = min(max(score, 0), 100.0)
    
    return {
        'score': float(score),
        'raw': float(incidence),
        'status': 'available'
    }
