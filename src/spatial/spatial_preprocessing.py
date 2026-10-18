"""Spatial preprocessing for clustering."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

def preprocess_for_clustering(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, List[str]]:
    """Isolate numeric features, handle missing data, and scale."""
    
    # 1. Separate metadata from numeric features
    metadata = df[exclude_cols] if exclude_cols else pd.DataFrame(index=df.index)
    numeric_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    features = df[numeric_cols].copy()
    
    # 2. Handle missing data (Simple mean imputation for spatial snapshot, or drop)
    # To keep simple: fill with mean
    features = features.fillna(features.mean())
    
    # 3. Scale
    scaler = StandardScaler()
    
    if not features.empty and len(features) > 0:
        scaled_array = scaler.fit_transform(features)
        scaled_df = pd.DataFrame(scaled_array, columns=features.columns, index=features.index)
    else:
        scaled_df = features
        
    return scaled_df, metadata, scaler, numeric_cols
