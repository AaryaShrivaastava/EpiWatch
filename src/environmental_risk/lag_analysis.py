"""Lag analysis."""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import List

def compute_lag_correlations(
    df: pd.DataFrame,
    target: str,
    variables: List[str],
    lags: List[int],
    region: str,
    disease: str
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations across lags."""
    results = []
    
    df = df.sort_values('date').copy()
    
    for var in variables:
        if var not in df.columns:
            continue
            
        for lag in lags:
            if lag == 0:
                shifted_var = df[var]
            else:
                shifted_var = df[var].shift(lag)
                
            # Drop NA
            valid_mask = ~shifted_var.isna() & ~df[target].isna()
            valid_x = shifted_var[valid_mask]
            valid_y = df.loc[valid_mask, target]
            
            sample_count = len(valid_x)
            
            if sample_count > 10:
                pearson_corr, _ = pearsonr(valid_x, valid_y)
                spearman_corr, _ = spearmanr(valid_x, valid_y)
            else:
                pearson_corr = np.nan
                spearman_corr = np.nan
                
            results.append({
                'region': region,
                'disease': disease,
                'environmental_variable': var,
                'lag': lag,
                'lag_unit': 'days',
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'sample_count': sample_count
            })
            
    return pd.DataFrame(results)
