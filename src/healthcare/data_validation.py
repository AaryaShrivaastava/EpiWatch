"""Data Validation for Healthcare Module."""
import pandas as pd
from typing import List

def validate_healthcare_data(hc_df: pd.DataFrame) -> List[str]:
    """Validate healthcare data for inconsistencies and missingness."""
    issues = []
    
    if hc_df.empty:
        return ["Healthcare dataset is empty."]
        
    # Check duplicates
    if hc_df.duplicated(subset=['date', 'region']).any():
        issues.append("Duplicate region-date records found.")
        
    # Check negatives
    for col in hc_df.select_dtypes(include=['number']).columns:
        if (hc_df[col] < 0).any():
            issues.append(f"Negative values found in {col}.")
            
    # Impossible ratios (if occupied and total beds were both present)
    if 'occupied_beds' in hc_df.columns and 'hospital_beds' in hc_df.columns:
        if (hc_df['occupied_beds'] > hc_df['hospital_beds']).any():
            issues.append("occupied_beds exceeds hospital_beds in some records.")
            
    if 'occupied_icu_beds' in hc_df.columns and 'icu_beds' in hc_df.columns:
        if (hc_df['occupied_icu_beds'] > hc_df['icu_beds']).any():
            issues.append("occupied_icu_beds exceeds icu_beds in some records.")
            
    return issues
