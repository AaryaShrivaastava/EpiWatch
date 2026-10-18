"""Regional feature building."""
import pandas as pd
import numpy as np
from typing import Dict, Optional

def build_regional_features(
    datasets: Dict[str, Optional[pd.DataFrame]], 
    regions: list,
    target_date: str = "latest"
) -> pd.DataFrame:
    """Build a unified region-level feature table."""
    
    # Base dataframe
    features = pd.DataFrame({'region': regions})
    
    # 1. Demographics
    demo_df = datasets.get('demographics')
    if demo_df is not None and 'region' in demo_df.columns:
        cols_to_merge = ['region', 'population_2011', 'population_density_per_sq_km', 'urbanization_percent']
        valid_cols = [c for c in cols_to_merge if c in demo_df.columns]
        features = features.merge(demo_df[valid_cols], on='region', how='left')
        
    # 2. Epidemiology (Latest or specific window)
    epi_df = datasets.get('epidemiology')
    if epi_df is not None and 'region' in epi_df.columns and 'date' in epi_df.columns:
        epi_df['date'] = pd.to_datetime(epi_df['date'])
        
        # Get latest stats per region
        epi_latest = epi_df.sort_values('date').groupby('region').tail(1)
        epi_cols = ['region', 'new_cases', 'active_cases', 'test_positivity_rate_percent']
        valid_cols = [c for c in epi_cols if c in epi_latest.columns]
        features = features.merge(epi_latest[valid_cols], on='region', how='left')
        
        # Calculate incidence per 100k
        if 'population_2011' in features.columns and 'new_cases' in features.columns:
            features['incidence_per_100k'] = (features['new_cases'] / features['population_2011']) * 100000
            features['incidence_per_100k'] = features['incidence_per_100k'].replace([np.inf, -np.inf], np.nan)
            
    # 3. Healthcare Stress
    stress_df = datasets.get('healthcare_stress')
    if stress_df is not None and 'region' in stress_df.columns:
        if 'date' in stress_df.columns:
            stress_df['date'] = pd.to_datetime(stress_df['date'])
            stress_latest = stress_df.sort_values('date').groupby('region').tail(1)
        else:
            stress_latest = stress_df
            
        cols = ['region', 'healthcare_stress_score']
        valid_cols = [c for c in cols if c in stress_latest.columns]
        features = features.merge(stress_latest[valid_cols], on='region', how='left')
        
    # 4. Healthcare Preparedness
    prep_df = datasets.get('healthcare_prep')
    if prep_df is not None and 'region' in prep_df.columns:
        prep_latest = prep_df.groupby('region').tail(1) if 'date' in prep_df.columns else prep_df
        cols = ['region', 'preparedness_score']
        valid_cols = [c for c in cols if c in prep_latest.columns]
        features = features.merge(prep_latest[valid_cols], on='region', how='left')
        
    # 5. Forecasts
    forecast_df = datasets.get('disease_forecasts')
    if forecast_df is not None and 'region' in forecast_df.columns:
        if 'date' in forecast_df.columns:
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            f_latest = forecast_df.sort_values('date').groupby('region').tail(1)
        else:
            f_latest = forecast_df
            
        cols = ['region', 'predicted_cases', 'forecast_surge_score']
        valid_cols = [c for c in cols if c in f_latest.columns]
        features = features.merge(f_latest[valid_cols], on='region', how='left')
        
    return features
