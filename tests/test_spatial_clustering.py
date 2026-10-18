"""Tests for spatial clustering module."""
import pandas as pd
import numpy as np
import pytest
from src.spatial.validation import validate_spatial_data, check_clustering_feasibility
from src.spatial.clustering import run_kmeans
from src.spatial.spatial_preprocessing import preprocess_for_clustering
from src.spatial.regional_feature_builder import build_regional_features
from src.spatial.cluster_profiling import profile_clusters

def test_validate_spatial_data_missing_demographics():
    datasets = {'epidemiology': pd.DataFrame()}
    regions, issues = validate_spatial_data(datasets)
    assert any("Demographics dataset is missing" in i for i in issues)

def test_check_clustering_feasibility():
    regions = ['A', 'B', 'C']
    can_cluster, msg = check_clustering_feasibility(regions, min_required=5)
    assert not can_cluster
    assert "Insufficient" in msg

def test_build_regional_features():
    demo = pd.DataFrame({'region': ['A', 'B'], 'population_2011': [1000, 2000]})
    epi = pd.DataFrame({
        'region': ['A', 'B'], 
        'date': ['2024-01-01', '2024-01-01'],
        'new_cases': [50, 20]
    })
    
    datasets = {'demographics': demo, 'epidemiology': epi}
    df = build_regional_features(datasets, ['A', 'B'])
    
    assert len(df) == 2
    # incidence A = 50 / 1000 * 100k = 5000
    assert np.isclose(df.loc[df['region'] == 'A', 'incidence_per_100k'].values[0], 5000)

def test_preprocess_for_clustering():
    df = pd.DataFrame({
        'region': ['A', 'B', 'C'],
        'incidence': [10, np.nan, 30]
    })
    
    scaled, meta, scaler, cols = preprocess_for_clustering(df, exclude_cols=['region'])
    
    # np.nan should be filled with mean (20)
    assert 'incidence' in scaled.columns
    assert 'region' not in scaled.columns
    assert not scaled['incidence'].isna().any()

def test_run_kmeans():
    features = pd.DataFrame({
        'val1': [0, 0, 10, 10],
        'val2': [0, 0, 10, 10]
    })
    
    labels, metrics = run_kmeans(features, n_clusters=2)
    assert len(labels) == 4
    # The two [0,0] should be in one cluster, [10,10] in another
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]

def test_run_kmeans_insufficient_data():
    features = pd.DataFrame({'val1': [0, 10]})
    with pytest.raises(ValueError):
        run_kmeans(features, n_clusters=3)
        
def test_profile_clusters():
    df = pd.DataFrame({
        'region': ['A', 'B', 'C', 'D'],
        'incidence_per_100k': [10, 20, 100, 120],
        'cluster': [0, 0, 1, 1]
    })
    
    profile = profile_clusters(df, 'cluster')
    assert len(profile) == 2
    assert profile.loc[profile['cluster'] == 0, 'region_count'].values[0] == 2
    assert "Lower" in profile.loc[profile['cluster'] == 0, 'description'].values[0]
    assert "Elevated" in profile.loc[profile['cluster'] == 1, 'description'].values[0]
