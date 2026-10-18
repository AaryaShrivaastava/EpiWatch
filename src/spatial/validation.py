"""Data Validation for Spatial Module."""
import pandas as pd
from typing import List, Tuple

def validate_spatial_data(datasets: dict) -> Tuple[List[str], List[str]]:
    """Validate loaded spatial datasets and region counts."""
    issues = []
    available_regions = []
    
    if datasets.get('demographics') is not None:
        demo_df = datasets['demographics']
        if 'region' in demo_df.columns:
            available_regions = demo_df['region'].unique().tolist()
            if demo_df.duplicated(subset=['region']).any():
                issues.append("Duplicate regions found in demographics.")
        else:
            issues.append("Missing 'region' column in demographics.")
    else:
        issues.append("Demographics dataset is missing.")
        
    for name, df in datasets.items():
        if df is None:
            issues.append(f"Optional or missing dataset: {name}")
        elif 'region' not in df.columns and name != 'boundaries':
            issues.append(f"Dataset {name} is missing 'region' column.")
            
    return available_regions, issues

def check_clustering_feasibility(regions: List[str], min_required: int = 5) -> Tuple[bool, str]:
    """Determine if clustering is possible."""
    n_regions = len(regions)
    if n_regions < min_required:
        return False, f"Insufficient geographical observations ({n_regions} regions). Minimum required is {min_required}."
    return True, "Sufficient regions for clustering."
