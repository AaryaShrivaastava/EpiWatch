"""Feature builder for environmental risk."""
import pandas as pd
from typing import List, Dict

def build_features(
    df: pd.DataFrame,
    variables: Dict[str, List[str]],
    lags: List[int],
    rolling_windows: List[int]
) -> pd.DataFrame:
    """Generate lagged and rolling features for environmental variables."""
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    for var, ops in variables.items():
        if var not in df.columns:
            continue
            
        # Lags
        if 'lag' in ops:
            for lag in lags:
                if lag > 0:
                    df[f'{var}_lag_{lag}'] = df[var].shift(lag)
                    
        # Rolling Mean
        if 'mean' in ops:
            for window in rolling_windows:
                df[f'{var}_{window}d_mean'] = df[var].rolling(window=window, min_periods=1).mean()
                
        # Rolling Sum
        if 'sum' in ops:
            for window in rolling_windows:
                df[f'{var}_{window}d_sum'] = df[var].rolling(window=window, min_periods=1).sum()
                
    return df
