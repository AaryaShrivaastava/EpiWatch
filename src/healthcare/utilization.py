"""Utilization metrics for Healthcare."""
import pandas as pd
import numpy as np

def estimate_occupied_beds(epi_df: pd.DataFrame, hc_df: pd.DataFrame, los_days: int = 7, icu_los: int = 10) -> pd.DataFrame:
    """
    Since actual occupied beds are missing in healthcare data,
    we estimate point-in-time occupancy from daily epidemiological admissions
    using a rolling sum window representing Length of Stay (LOS).
    """
    epi = epi_df.copy().set_index('date').sort_index()
    
    if 'hospitalizations' in epi.columns:
        # Sum of hospitalizations over last 'los_days' days
        epi['estimated_occupied_beds'] = epi['hospitalizations'].rolling(window=los_days, min_periods=1).sum()
    else:
        epi['estimated_occupied_beds'] = np.nan
        
    if 'icu_admissions' in epi.columns:
        epi['estimated_occupied_icu_beds'] = epi['icu_admissions'].rolling(window=icu_los, min_periods=1).sum()
    else:
        epi['estimated_occupied_icu_beds'] = np.nan
        
    # Merge estimates into healthcare dataset by resample to monthly or just forward filling daily
    # Since hc_df is monthly, we will interpolate or merge as of month start/end.
    hc = hc_df.copy().set_index('date').sort_index()
    
    # We will resample epi to monthly mean or max to join with hc, or join directly if hc is daily
    # For now, let's join daily epi with forward filled hc capacity
    daily_hc = hc.resample('D').ffill()
    
    # Merge daily epi occupancy with daily capacity
    merged = epi[['estimated_occupied_beds', 'estimated_occupied_icu_beds']].join(daily_hc, how='inner')
    
    return merged.reset_index()

def calculate_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate utilization ratios and percentages."""
    df = df.copy()
    
    if 'estimated_occupied_beds' in df.columns and 'hospital_beds' in df.columns:
        df['bed_occupancy_ratio'] = df['estimated_occupied_beds'] / df['hospital_beds']
        # Handle div by zero or missing
        df['bed_occupancy_ratio'] = df['bed_occupancy_ratio'].replace([np.inf, -np.inf], np.nan)
        df['bed_occupancy_percent'] = df['bed_occupancy_ratio'] * 100.0
    else:
        df['bed_occupancy_ratio'] = np.nan
        df['bed_occupancy_percent'] = np.nan
        
    if 'estimated_occupied_icu_beds' in df.columns and 'icu_beds' in df.columns:
        df['icu_occupancy_ratio'] = df['estimated_occupied_icu_beds'] / df['icu_beds']
        df['icu_occupancy_ratio'] = df['icu_occupancy_ratio'].replace([np.inf, -np.inf], np.nan)
        df['icu_occupancy_percent'] = df['icu_occupancy_ratio'] * 100.0
    else:
        df['icu_occupancy_ratio'] = np.nan
        df['icu_occupancy_percent'] = np.nan
        
    if 'tests_performed' in df.columns and 'testing_capacity_tests_per_day' in df.columns:
        df['testing_utilization_ratio'] = df['tests_performed'] / df['testing_capacity_tests_per_day']
        df['testing_utilization_percent'] = df['testing_utilization_ratio'] * 100.0
        
    return df
