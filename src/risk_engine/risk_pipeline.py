"""Main Risk Engine Pipeline."""
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from src.risk_engine.config import get_risk_config
from src.risk_engine.input_registry import load_risk_data, validate_risk_data
from src.risk_engine.disease_burden import assess_disease_burden
from src.risk_engine.outbreak_assessment import assess_outbreak_signal
from src.risk_engine.forecast_assessment import assess_forecast_trajectory
from src.risk_engine.environmental_assessment import assess_environmental
from src.risk_engine.healthcare_assessment import assess_healthcare
from src.risk_engine.scoring import calculate_integrated_score, classify_risk
from src.risk_engine.risk_explanation import generate_risk_explanation

logger = logging.getLogger(__name__)

def run_regional_risk_assessment():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Regional Risk Assessment Engine (Phase 9)")
    
    config = get_risk_config()
    region = config.get("risk_engine.region", "Bhopal")
    
    # Dirs
    out_dir = Path(config.get("risk_engine.outputs.analytics_dir", "data/risk_engine"))
    reports_dir = Path(config.get("risk_engine.outputs.reports_dir", "reports/risk_engine"))
    plots_dir = Path(config.get("risk_engine.outputs.plots_dir", "reports/risk_engine/plots"))
    for d in [out_dir, reports_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 1. Load Data
    paths = {
        'epi_path': config.get("risk_engine.data.epi_path"),
        'demo_path': config.get("risk_engine.data.demographics_path"),
        'outbreak_path': config.get("risk_engine.data.outbreak_path"),
        'forecast_path': config.get("risk_engine.data.forecast_path"),
        'env_path': config.get("risk_engine.data.environmental_path"),
        'hc_stress_path': config.get("risk_engine.data.healthcare_stress_path"),
        'hc_prep_path': config.get("risk_engine.data.healthcare_prep_path")
    }
    
    datasets = load_risk_data(paths)
    is_valid, issues = validate_risk_data(datasets)
    
    if not is_valid:
        logger.error(f"Validation failed: {issues}")
        
    # 2. Assess Dimensions
    assessments = {
        'disease_burden': assess_disease_burden(datasets.get('epi_path'), datasets.get('demo_path'), region),
        'outbreak_signal': assess_outbreak_signal(datasets.get('outbreak_path'), region),
        'forecast_trajectory': assess_forecast_trajectory(datasets.get('forecast_path'), region),
        'environmental_risk': assess_environmental(datasets.get('env_path'), region)
    }
    
    hc = assess_healthcare(datasets.get('hc_stress_path'), datasets.get('hc_prep_path'), region)
    assessments['healthcare_stress'] = hc['stress']
    
    # 3. Scoring
    weights = config.get("risk_engine.scoring.weights", {})
    integrated_score, contribs = calculate_integrated_score(assessments, weights)
    
    thresholds = config.get("risk_engine.scoring.categories", {})
    category = classify_risk(integrated_score, thresholds)
    
    # 4. Explanation
    explanation = generate_risk_explanation(region, category, integrated_score, assessments, contribs)
    
    # 5. Outputs
    res_df = pd.DataFrame([{
        'region': region,
        'integrated_regional_risk_score': integrated_score,
        'disease_risk_score': assessments['disease_burden']['score'],
        'healthcare_stress_score': hc['stress']['score'],
        'healthcare_preparedness_score': hc['preparedness']['score'],
        'risk_category': category
    }])
    res_df.to_csv(out_dir / "regional_risk_assessments.csv", index=False)
    
    contrib_list = []
    for k, v in assessments.items():
        contrib_list.append({
            'region': region,
            'dimension': k,
            'raw_value': v['raw'],
            'normalized_score': v['score'],
            'status': v['status'],
            'configured_weight': weights.get(k, 0),
            'numerical_contribution': contribs.get(k, 0)
        })
    pd.DataFrame(contrib_list).to_csv(out_dir / "risk_contributions.csv", index=False)
    
    with open(reports_dir / "regional_risk_report.md", "w") as f:
        f.write("# Regional Risk Assessment Report\n\n")
        f.write(explanation)
        f.write(f"\n\n## Scores\nIntegrated: {integrated_score}\nDisease Burden: {assessments['disease_burden']['score']}")
        f.write(f"\nHealthcare Stress: {hc['stress']['score']}")
        
    # Visualization (simple bar chart of contribs)
    if contribs:
        plt.figure(figsize=(8,5))
        plt.bar(contribs.keys(), contribs.values(), color='coral')
        plt.xticks(rotation=45)
        plt.ylabel('Numerical Contribution')
        plt.title(f'Risk Score Contributions - {region}')
        plt.tight_layout()
        plt.savefig(plots_dir / "risk_dimension_contribution.png")
        plt.close()
        
    logger.info("Risk Engine Pipeline completed.")

if __name__ == "__main__":
    run_regional_risk_assessment()
