"""Stress score and capacity analysis."""
import pandas as pd
import numpy as np

def calculate_stress_score(
    df: pd.DataFrame, 
    weights: dict, 
    thresholds: dict
) -> pd.DataFrame:
    """Calculate the normalized healthcare stress score."""
    df = df.copy()
    
    score = np.zeros(len(df))
    total_weight = 0
    available_inds = []
    missing_inds = []
    
    if 'bed_occupancy' in weights:
        if 'bed_occupancy_ratio' in df.columns:
            # Multiply by 100 to get a 0-100 scale (capping at 100 for purely stress metric, but can go over if >100%)
            val = df['bed_occupancy_ratio'].fillna(0) * 100.0
            val = val.clip(0, 100)
            score += val * weights['bed_occupancy']
            total_weight += weights['bed_occupancy']
            available_inds.append('bed_occupancy')
        else:
            missing_inds.append('bed_occupancy')
            
    if 'icu_occupancy' in weights:
        if 'icu_occupancy_ratio' in df.columns:
            val = df['icu_occupancy_ratio'].fillna(0) * 100.0
            val = val.clip(0, 100)
            score += val * weights['icu_occupancy']
            total_weight += weights['icu_occupancy']
            available_inds.append('icu_occupancy')
        else:
            missing_inds.append('icu_occupancy')
            
    if 'oxygen_utilization' in weights:
        if 'available_oxygen_percent' in df.columns:
            # High utilization = 100 - available
            val = 100.0 - df['available_oxygen_percent'].fillna(100)
            val = val.clip(0, 100)
            score += val * weights['oxygen_utilization']
            total_weight += weights['oxygen_utilization']
            available_inds.append('oxygen_utilization')
        else:
            missing_inds.append('oxygen_utilization')
            
    # Calculate final score
    if total_weight > 0:
        final_score = score / total_weight
        # Re-mask missing regions if the denominator was essentially entirely missing
        # But we filled NAs with 0 above. Let's strictly only score if we have AT LEAST 1 indicator
        mask = (len(available_inds) > 0)
        df['healthcare_stress_score'] = np.where(mask, final_score, np.nan)
    else:
        df['healthcare_stress_score'] = np.nan
        
    def categorize(s):
        if pd.isna(s):
            return "UNAVAILABLE"
        if s < thresholds.get('moderate', 40): return "LOW_STRESS"
        if s < thresholds.get('high', 60): return "MODERATE_STRESS"
        if s < thresholds.get('critical', 80): return "HIGH_STRESS"
        return "CRITICAL_STRESS"
        
    df['stress_category'] = df['healthcare_stress_score'].apply(categorize)
    df['available_stress_indicators'] = ",".join(available_inds)
    df['missing_stress_indicators'] = ",".join(missing_inds)
    
    return df
