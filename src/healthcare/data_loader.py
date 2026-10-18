"""Data Loading for Healthcare Module."""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

def load_healthcare_data(
    hc_path: Path, epi_path: Path, demo_path: Path, forecast_path: Path,
    region: str, disease: str
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Optional[pd.DataFrame]]:
    """Load healthcare, epidemiological, demographics, and forecast data."""
    
    # Load and filter demographics
    demo_df = pd.read_csv(demo_path)
    demo_region = demo_df[demo_df['region'] == region]
    if demo_region.empty:
        raise ValueError(f"Region {region} not found in demographics.")
    
    population = float(demo_region.iloc[0]['population_2011'])
    
    # Load and filter healthcare
    hc_df = pd.read_csv(hc_path)
    hc_df = hc_df[hc_df['region'] == region].copy()
    
    # Load and filter epi
    epi_df = pd.read_csv(epi_path)
    epi_df = epi_df[(epi_df['region'] == region) & (epi_df['disease'] == disease)].copy()
    
    # Load forecasts if available
    forecast_df = None
    if forecast_path.exists():
        f_df = pd.read_csv(forecast_path)
        f_df = f_df[(f_df['region'] == region) & (f_df['disease'] == disease)].copy()
        if not f_df.empty:
            forecast_df = f_df
            
    # Standardize dates
    if 'month' in hc_df.columns and 'date' not in hc_df.columns:
        hc_df['date'] = pd.to_datetime(hc_df['month'])
    else:
        hc_df['date'] = pd.to_datetime(hc_df['date'])
        
    epi_df['date'] = pd.to_datetime(epi_df['date'])
    if forecast_df is not None:
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
    return hc_df.sort_values('date'), epi_df.sort_values('date'), population, forecast_df
