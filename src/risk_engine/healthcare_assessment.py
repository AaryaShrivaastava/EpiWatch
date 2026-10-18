"""Healthcare assessment."""
import pandas as pd
import numpy as np
from typing import Dict, Any

def assess_healthcare(hc_stress_df: pd.DataFrame, hc_prep_df: pd.DataFrame, region: str) -> Dict[str, Dict[str, Any]]:
    """Extract healthcare stress and preparedness."""
    res = {
        'stress': {'score': np.nan, 'raw': np.nan, 'status': 'missing'},
        'preparedness': {'score': np.nan, 'raw': np.nan, 'status': 'missing'}
    }
    
    if hc_stress_df is not None and not hc_stress_df.empty:
        df = hc_stress_df[hc_stress_df['region'] == region]
        if not df.empty:
            if 'date' in df.columns:
                df = df.sort_values('date')
            latest = df.iloc[-1]
            if 'healthcare_stress_score' in latest and not pd.isna(latest['healthcare_stress_score']):
                score = float(latest['healthcare_stress_score'])
                res['stress'] = {'score': score, 'raw': score, 'status': 'available'}
                
    if hc_prep_df is not None and not hc_prep_df.empty:
        df = hc_prep_df[hc_prep_df['region'] == region]
        if not df.empty:
            latest = df.iloc[-1]
            if 'preparedness_score' in latest and not pd.isna(latest['preparedness_score']):
                score = float(latest['preparedness_score'])
                res['preparedness'] = {'score': score, 'raw': score, 'status': 'available'}
                
    return res
