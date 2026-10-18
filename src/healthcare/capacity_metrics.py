"""Capacity metrics for Healthcare."""
import pandas as pd
import numpy as np

def calculate_capacity_metrics(hc_df: pd.DataFrame, population: float) -> pd.DataFrame:
    """Calculate normalized capacity metrics per 100k population."""
    df = hc_df.copy()
    
    if population <= 0:
        # Avoid division by zero
        df['beds_per_100k'] = np.nan
        df['icu_beds_per_100k'] = np.nan
        df['doctors_per_100k'] = np.nan
        return df
        
    if 'hospital_beds' in df.columns:
        df['beds_per_100k'] = df['hospital_beds'] / population * 100000
    else:
        df['beds_per_100k'] = np.nan
        
    if 'icu_beds' in df.columns:
        df['icu_beds_per_100k'] = df['icu_beds'] / population * 100000
    else:
        df['icu_beds_per_100k'] = np.nan
        
    if 'doctors' in df.columns:
        df['doctors_per_100k'] = df['doctors'] / population * 100000
    else:
        df['doctors_per_100k'] = np.nan
        
    # Nurses are not natively in our dataset right now, but check just in case
    if 'nurses' in df.columns:
        df['nurses_per_100k'] = df['nurses'] / population * 100000
        
    return df
