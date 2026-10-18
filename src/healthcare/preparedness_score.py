"""Preparedness scoring."""
import pandas as pd
import numpy as np

def calculate_preparedness_score(
    df: pd.DataFrame, 
    weights: dict, 
    benchmarks: dict
) -> pd.DataFrame:
    """Calculate long-term structural preparedness."""
    df = df.copy()
    
    score = np.zeros(len(df))
    total_weight = 0
    available_inds = []
    missing_inds = []
    
    if 'beds_per_100k' in weights:
        if 'beds_per_100k' in df.columns:
            # Normalize to benchmark
            val = (df['beds_per_100k'].fillna(0) / benchmarks.get('beds_per_100k', 300)) * 100.0
            val = val.clip(0, 100)
            score += val * weights['beds_per_100k']
            total_weight += weights['beds_per_100k']
            available_inds.append('beds_per_100k')
        else:
            missing_inds.append('beds_per_100k')
            
    if 'icu_beds_per_100k' in weights:
        if 'icu_beds_per_100k' in df.columns:
            val = (df['icu_beds_per_100k'].fillna(0) / benchmarks.get('icu_beds_per_100k', 30)) * 100.0
            val = val.clip(0, 100)
            score += val * weights['icu_beds_per_100k']
            total_weight += weights['icu_beds_per_100k']
            available_inds.append('icu_beds_per_100k')
        else:
            missing_inds.append('icu_beds_per_100k')
            
    if 'doctors_per_100k' in weights:
        if 'doctors_per_100k' in df.columns:
            val = (df['doctors_per_100k'].fillna(0) / benchmarks.get('doctors_per_100k', 100)) * 100.0
            val = val.clip(0, 100)
            score += val * weights['doctors_per_100k']
            total_weight += weights['doctors_per_100k']
            available_inds.append('doctors_per_100k')
        else:
            missing_inds.append('doctors_per_100k')
            
    if 'medicine_availability' in weights:
        if 'medicine_availability_percent' in df.columns:
            val = df['medicine_availability_percent'].fillna(0)
            score += val * weights['medicine_availability']
            total_weight += weights['medicine_availability']
            available_inds.append('medicine_availability')
        else:
            missing_inds.append('medicine_availability')
            
    if total_weight > 0:
        final_score = score / total_weight
        df['preparedness_score'] = np.where(len(available_inds) > 0, final_score, np.nan)
    else:
        df['preparedness_score'] = np.nan
        
    df['available_preparedness_indicators'] = ",".join(available_inds)
    df['missing_preparedness_indicators'] = ",".join(missing_inds)
    df['preparedness_method'] = "benchmark_weighted"
    
    return df
