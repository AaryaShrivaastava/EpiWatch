"""Cluster profiling and logic."""
import pandas as pd

def profile_clusters(features_df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Generate profiles for each cluster based on available features."""
    
    if cluster_col not in features_df.columns:
        return pd.DataFrame()
        
    numeric_cols = [c for c in features_df.columns if pd.api.types.is_numeric_dtype(features_df[c]) and c != cluster_col]
    
    # Calculate means for each numeric column by cluster
    profile = features_df.groupby(cluster_col)[numeric_cols].mean().reset_index()
    
    # Add count
    counts = features_df.groupby(cluster_col).size().reset_index(name='region_count')
    profile = profile.merge(counts, on=cluster_col)
    
    # Generate labels (if possible)
    def assign_label(row):
        # Extremely basic example: if incidence is high, call it high risk
        if 'incidence_per_100k' in row:
            if row['incidence_per_100k'] > 50:
                return "Elevated disease burden"
            else:
                return "Lower observed disease burden"
        return "Profiled Cluster"
        
    profile['description'] = profile.apply(assign_label, axis=1)
    
    return profile
