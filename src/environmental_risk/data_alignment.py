"""Data alignment for environmental risk."""
import pandas as pd

def align_datasets(env_df: pd.DataFrame, epi_df: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
    """Align environmental and epidemiological datasets."""
    
    # Verify unique region-date
    if env_df.duplicated(['date']).any():
        env_df = env_df.drop_duplicates(['date']) # basic deduplication
        
    if epi_df.duplicated(['date']).any():
        epi_df = epi_df.drop_duplicates(['date'])
        
    env_df = env_df.set_index('date')
    epi_df = epi_df.set_index('date')
    
    # If frequency is weekly, resample.
    if frequency == 'W':
        env_agg = {
            'temperature_c': 'mean',
            'rainfall_mm': 'sum',
            'humidity_percent': 'mean',
            'air_quality_index': 'mean',
            'water_accumulation_index': 'mean'
        }
        # Only aggregate existing columns
        env_agg = {k: v for k, v in env_agg.items() if k in env_df.columns}
        env_df = env_df.resample('W').agg(env_agg)
        
        epi_agg = {
            'new_cases': 'sum',
            'active_cases': 'mean',
            'hospitalizations': 'sum'
        }
        epi_agg = {k: v for k, v in epi_agg.items() if k in epi_df.columns}
        epi_df = epi_df.resample('W').agg(epi_agg)
        
    # Drop overlapping columns from env_df
    overlap = set(env_df.columns).intersection(set(epi_df.columns))
    env_df = env_df.drop(columns=list(overlap))
    
    # Merge
    aligned = epi_df.join(env_df, how='inner')
    
    # Check if anything dropped
    aligned = aligned.reset_index()
    
    return aligned
