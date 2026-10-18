"""Spatial clustering algorithms."""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from typing import Tuple, Optional

def run_kmeans(
    features: pd.DataFrame, 
    n_clusters: int, 
    random_state: int = 42
) -> Tuple[np.ndarray, dict]:
    """Run KMeans clustering."""
    if len(features) < n_clusters:
        raise ValueError("Number of regions is less than number of clusters.")
        
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = model.fit_predict(features)
    
    metrics = {
        'inertia': model.inertia_
    }
    return labels, metrics

def run_dbscan(
    features: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 3
) -> Tuple[np.ndarray, dict]:
    """Run DBSCAN clustering."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)
    
    n_noise = list(labels).count(-1)
    metrics = {
        'noise_points': n_noise
    }
    return labels, metrics
