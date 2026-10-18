"""Recommendation Pipeline."""
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

from src.recommendations.config import get_recommendation_config
from src.recommendations.input_registry import load_recommendation_data
from src.recommendations.evidence_builder import extract_evidence
from src.recommendations.rule_engine import evaluate_rules

logger = logging.getLogger(__name__)

def run_recommendation_pipeline():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Evidence-Based Public-Health Recommendation Engine (Phase 11)")
    
    config = get_recommendation_config()
    region = config.get("recommendations.region", "Bhopal")
    
    # Dirs
    out_dir = Path(config.get("recommendations.outputs.data_dir", "data/recommendations"))
    reports_dir = Path(config.get("recommendations.outputs.reports_dir", "reports/recommendations"))
    plots_dir = Path(config.get("recommendations.outputs.plots_dir", "reports/recommendations/plots"))
    
    for d in [out_dir, reports_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Load Data
    paths = {
        'risk_assessment_path': config.get("recommendations.data.risk_assessment_path"),
        'outbreak_path': config.get("recommendations.data.outbreak_path"),
        'forecast_path': config.get("recommendations.data.forecast_path"),
        'env_path': config.get("recommendations.data.env_path"),
        'hc_stress_path': config.get("recommendations.data.hc_stress_path"),
        'hc_prep_path': config.get("recommendations.data.hc_prep_path"),
        'resource_gap_path': config.get("recommendations.data.resource_gap_path")
    }
    
    datasets = load_recommendation_data(paths)
    
    # Build Evidence
    evidence = extract_evidence(datasets, region)
    
    # Evaluate Rules
    thresholds = config.get("recommendations.rules", {})
    recs = evaluate_rules(evidence, thresholds, region)
    
    # Output to CSV
    if recs:
        df_recs = pd.DataFrame(recs)
        df_recs.to_csv(out_dir / "recommendations.csv", index=False)
        
        # Save Evidence mapping
        ev_df = pd.DataFrame([{'region': region, 'feature': k, 'value': v} for k, v in evidence.items()])
        ev_df.to_csv(out_dir / "recommendation_evidence.csv", index=False)
        
        # Report Generation
        with open(reports_dir / "recommendation_report.md", "w") as f:
            f.write("# Public-Health Recommendation Report\n\n")
            f.write(f"**Region:** {region}\n\n")
            f.write("## Executive Summary\n")
            f.write(f"Generated {len(recs)} evidence-based recommendations for review.\n\n")
            
            f.write("## Recommendations by Priority\n")
            for r in sorted(recs, key=lambda x: x['priority']):
                f.write(f"### [{r['priority']}] {r['title']}\n")
                f.write(f"- **Category**: {r['category']}\n")
                f.write(f"- **Stakeholder**: {r['stakeholder']}\n")
                f.write(f"- **Action**: {r['description']}\n")
                f.write(f"- **Evidence**: {r['rationale']}\n")
                f.write(f"- **Limitation**: {r['limitation']}\n\n")
                
        # Visualization
        counts = Counter([r['priority'] for r in recs])
        plt.figure(figsize=(8,5))
        plt.bar(counts.keys(), counts.values(), color='steelblue')
        plt.ylabel("Count")
        plt.title("Recommendations by Priority")
        plt.tight_layout()
        plt.savefig(plots_dir / "recommendation_priority.png")
        plt.close()
        
        counts_cat = Counter([r['category'] for r in recs])
        plt.figure(figsize=(10,6))
        plt.bar(counts_cat.keys(), counts_cat.values(), color='mediumseagreen')
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.title("Recommendations by Category")
        plt.tight_layout()
        plt.savefig(plots_dir / "recommendation_category.png")
        plt.close()
        
        logger.info(f"Generated {len(recs)} recommendations.")
    else:
        logger.info("No recommendations generated based on current evidence.")
        
    logger.info("Recommendation Pipeline completed.")

if __name__ == "__main__":
    run_recommendation_pipeline()
