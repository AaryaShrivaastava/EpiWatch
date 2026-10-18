"""Spatial Analysis Pipeline."""
import logging
import json
import pandas as pd
from pathlib import Path

from src.spatial.config import get_spatial_config
from src.spatial.data_loader import load_spatial_data
from src.spatial.validation import validate_spatial_data, check_clustering_feasibility
from src.spatial.regional_feature_builder import build_regional_features
from src.spatial.spatial_preprocessing import preprocess_for_clustering
from src.spatial.clustering import run_kmeans, run_dbscan
from src.spatial.cluster_profiling import profile_clusters
from src.spatial.hotspot_detection import detect_hotspots
from src.spatial.visualization import plot_incidence_comparison, plot_stress_vs_incidence

logger = logging.getLogger(__name__)

def run_spatial_analysis():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Spatial Risk & Regional Clustering Engine (Phase 8)")
    
    config = get_spatial_config()
    
    # Setup dirs
    out_dir = Path(config.get("spatial.outputs.features_dir", "data/spatial"))
    reports_dir = Path(config.get("spatial.outputs.reports_dir", "reports/spatial"))
    plots_dir = Path(config.get("spatial.outputs.plots_dir", "reports/spatial/plots"))
    
    for d in [out_dir, reports_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 1. Load Data
    paths = {
        'demographics': config.get("spatial.data.demographics_path"),
        'epidemiology': config.get("spatial.data.epidemiological_path"),
        'environmental': config.get("spatial.data.environmental_path"),
        'healthcare_stress': config.get("spatial.data.healthcare_stress_path"),
        'healthcare_prep': config.get("spatial.data.healthcare_prep_path"),
        'disease_forecasts': config.get("spatial.data.disease_forecasts_path"),
        'boundaries': config.get("spatial.data.boundary_file_path")
    }
    
    datasets = load_spatial_data(paths)
    
    # 2. Validate
    regions, issues = validate_spatial_data(datasets)
    
    # 3. Build Features
    features_df = build_regional_features(datasets, regions)
    features_df.to_csv(out_dir / "region_risk_features.csv", index=False)
    
    report_content = [
        "# Spatial Risk & Regional Clustering Report",
        "\n## Executive Summary",
        "This report summarizes regional epidemiological conditions, clusters, and hotspots."
    ]
    
    report_content.append("\n## Data Validation")
    if issues:
        report_content.append("Issues detected:")
        for issue in issues:
            report_content.append(f"- {issue}")
    else:
        report_content.append("No critical validation issues.")
        
    # 4. Clustering Feasibility
    min_regions = config.get("spatial.clustering.min_regions_required", 5)
    can_cluster, cluster_msg = check_clustering_feasibility(regions, min_required=min_regions)
    report_content.append(f"\n## Clustering\nStatus: {cluster_msg}")
    
    if can_cluster:
        # Preprocess
        exclude = ['region']
        scaled_df, meta, scaler, num_cols = preprocess_for_clustering(features_df, exclude)
        
        # KMeans
        n_clusters = config.get("spatial.clustering.n_clusters", 3)
        labels, metrics = run_kmeans(scaled_df, n_clusters=n_clusters)
        
        features_df['cluster'] = labels
        features_df['clustering_method'] = 'kmeans'
        
        features_df[['region', 'cluster', 'clustering_method']].to_csv(out_dir / "cluster_assignments.csv", index=False)
        
        # Profile
        profile_df = profile_clusters(features_df, 'cluster')
        profile_df.to_csv(out_dir / "cluster_profiles.csv", index=False)
        
        report_content.append(f"Method: KMeans (k={n_clusters})")
        report_content.append(f"Inertia: {metrics.get('inertia', 0):.2f}")
        report_content.append("\nProfiles:\n" + profile_df[['cluster', 'region_count', 'description']].to_markdown())
        
    # 5. Hotspot Detection
    min_hotspot = config.get("spatial.hotspot.min_regions_required", 10)
    boundaries = datasets.get('boundaries')
    has_hotspots, hs_msg, hs_df = detect_hotspots(regions, min_required=min_hotspot, boundary_data=boundaries)
    
    report_content.append(f"\n## Hotspot Detection\nStatus: {hs_msg}")
    if has_hotspots and not hs_df.empty:
        hs_df.to_csv(out_dir / "spatial_hotspots.csv", index=False)
        
    # 6. Visualizations (Even if boundaries missing)
    plot_incidence_comparison(features_df, plots_dir / "regional_incidence.png")
    plot_stress_vs_incidence(features_df, plots_dir / "stress_vs_incidence.png")
    
    report_content.append("\n## Geographic Visualization\nBoundary files missing. Map generation skipped. Non-map visualizations (bar charts, scatter plots) have been generated.")
    
    # Save Report
    with open(reports_dir / "spatial_analysis_report.md", "w") as f:
        f.write("\n".join(report_content))
        
    logger.info("Spatial Pipeline completed.")

if __name__ == "__main__":
    run_spatial_analysis()
