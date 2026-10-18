"""Pipeline for Phase 7: Healthcare Preparedness & Stress Modeling."""
import logging
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from .config import get_hc_config
from .data_loader import load_healthcare_data
from .data_validation import validate_healthcare_data
from .capacity_metrics import calculate_capacity_metrics
from .utilization import estimate_occupied_beds, calculate_utilization
from .demand_estimation import estimate_future_demand
from .stress_model import calculate_stress_score
from .preparedness_score import calculate_preparedness_score
from .resource_gap import calculate_resource_gaps
from .explanation import generate_explanations

logger = logging.getLogger(__name__)

def run_pipeline():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Healthcare Preparedness & Stress Modeling (Phase 7)")
    
    config = get_hc_config()
    
    region = config.get("healthcare.region", "Bhopal")
    disease = config.get("healthcare.disease", "COVID-19")
    
    hc_path = Path(config.get("healthcare.data.healthcare_path"))
    epi_path = Path(config.get("healthcare.data.epidemiological_path"))
    demo_path = Path(config.get("healthcare.data.demographics_path"))
    forecast_path = Path(config.get("healthcare.data.forecast_path"))
    
    out_dir = Path(config.get("healthcare.outputs.analytics_dir"))
    pred_dir = Path(config.get("healthcare.outputs.predictions_dir"))
    reports_dir = Path(config.get("healthcare.outputs.reports_dir"))
    plots_dir = Path(config.get("healthcare.outputs.plots_dir"))
    
    for d in [out_dir, pred_dir, reports_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Load data
    hc_df, epi_df, pop, forecast_df = load_healthcare_data(
        hc_path, epi_path, demo_path, forecast_path, region, disease
    )
    
    # Validation
    issues = validate_healthcare_data(hc_df)
    with open(reports_dir / "data_quality_report.md", "w") as f:
        f.write("# Healthcare Data Quality Report\n\n")
        if issues:
            for issue in issues:
                f.write(f"- {issue}\n")
        else:
            f.write("- No critical issues found.\n")
            
    # Capacity Metrics
    capacity_df = calculate_capacity_metrics(hc_df, pop)
    capacity_df.to_csv(out_dir / "capacity_metrics.csv", index=False)
    
    # Utilization
    los = config.get("healthcare.assumptions.hospital_los_days", 7)
    icu_los = config.get("healthcare.assumptions.icu_los_days", 10)
    
    occ_df = estimate_occupied_beds(epi_df, capacity_df, los, icu_los)
    util_df = calculate_utilization(occ_df)
    util_df.to_csv(out_dir / "utilization_metrics.csv", index=False)
    
    # Stress Score
    s_weights = config.get("healthcare.scoring.stress_weights", {})
    s_thresh = config.get("healthcare.scoring.stress_thresholds", {})
    stress_df = calculate_stress_score(util_df, s_weights, s_thresh)
    stress_df.to_csv(out_dir / "healthcare_stress.csv", index=False)
    
    # Preparedness Score
    p_weights = config.get("healthcare.scoring.preparedness_weights", {})
    p_bench = config.get("healthcare.scoring.preparedness_benchmarks", {})
    prep_df = calculate_preparedness_score(capacity_df, p_weights, p_bench)
    prep_df['data_provenance'] = 'SOURCE_CALIBRATED_SYNTHETIC'
    prep_df.to_csv(out_dir / "healthcare_preparedness.csv", index=False)
    
    # Demand Estimation
    h_rate = config.get("healthcare.assumptions.forecast_hospitalization_rate", 0.1)
    i_rate = config.get("healthcare.assumptions.forecast_icu_rate", 0.02)
    demand_df = estimate_future_demand(forecast_df, h_rate, i_rate, los, icu_los)
    if demand_df is not None:
        demand_df.to_csv(pred_dir / "healthcare_demand.csv", index=False)
        
    # Resource Gap Analysis
    gap_df = calculate_resource_gaps(capacity_df, demand_df)
    if not gap_df.empty:
        gap_df.to_csv(out_dir / "resource_gap_analysis.csv", index=False)
        
    # Explanation
    explanation = generate_explanations(stress_df, region)
    
    # Report
    with open(reports_dir / "healthcare_report.md", "w") as f:
        f.write("# Healthcare Preparedness & Stress Report\n\n")
        f.write(explanation)
        
    # Visualizations
    plt.figure(figsize=(10,6))
    if 'hospital_beds' in util_df.columns and 'estimated_occupied_beds' in util_df.columns:
        plt.plot(util_df['date'], util_df['hospital_beds'], label='Total Beds')
        plt.plot(util_df['date'], util_df['estimated_occupied_beds'], label='Occupied Beds')
        plt.legend()
        plt.title('Bed Capacity vs Occupied Beds')
        plt.savefig(plots_dir / "beds_vs_occupied.png")
    plt.close()
    
    plt.figure(figsize=(10,6))
    if 'icu_beds' in util_df.columns and 'estimated_occupied_icu_beds' in util_df.columns:
        plt.plot(util_df['date'], util_df['icu_beds'], label='Total ICU Beds')
        plt.plot(util_df['date'], util_df['estimated_occupied_icu_beds'], label='Occupied ICU Beds')
        plt.legend()
        plt.title('ICU Capacity vs Occupied ICU Beds')
        plt.savefig(plots_dir / "icu_vs_occupied.png")
    plt.close()
    
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
