"""Data loading for environmental risk."""
import pandas as pd
from pathlib import Path
from typing import Tuple

def load_data(env_path: Path, epi_path: Path, region: str, disease: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and filter environmental and epidemiological datasets."""
    
    if not env_path.exists():
        raise FileNotFoundError(f"Environmental data not found at {env_path}")
    if not epi_path.exists():
        raise FileNotFoundError(f"Epidemiological data not found at {epi_path}")
        
    env_df = pd.read_csv(env_path)
    epi_df = pd.read_csv(epi_path)
    
    # Filter by region
    if 'region' in env_df.columns:
        env_df = env_df[env_df['region'] == region].copy()
    if 'region' in epi_df.columns:
        epi_df = epi_df[epi_df['region'] == region].copy()
        
    # Filter by disease for epi
    if 'disease' in epi_df.columns:
        epi_df = epi_df[epi_df['disease'] == disease].copy()
        
    # Convert dates
    env_df['date'] = pd.to_datetime(env_df['date'])
    epi_df['date'] = pd.to_datetime(epi_df['date'])
    
    return env_df.sort_values('date'), epi_df.sort_values('date')
