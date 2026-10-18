"""Pipeline for Environmental Risk."""
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.environmental_risk.config import get_env_config
from src.environmental_risk.data_loader import load_data
from src.environmental_risk.data_alignment import align_datasets
from src.environmental_risk.feature_builder import build_features
from src.environmental_risk.lag_analysis import compute_lag_correlations
from src.environmental_risk.statistical_analysis import fit_negative_binomial_model
from src.environmental_risk.risk_model import train_evaluate_ml_model
from src.environmental_risk.evaluation import evaluate_predictions
from src.environmental_risk.risk_scoring import generate_risk_score
from src.environmental_risk.explanation import generate_explanations

logger = logging.getLogger(__name__)

def run_pipeline():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Environmental Risk Pipeline (Phase 6)")
    
    config = get_env_config()
    
    disease = config.get("environmental_risk.disease", "Dengue")
    region = config.get("environmental_risk.region", "Bhopal")
    
    env_path = Path(config.get("environmental_risk.data.environmental_path"))
    epi_path = Path(config.get("environmental_risk.data.epidemiological_path"))
    
    out_dir = Path(config.get("environmental_risk.outputs.data_dir"))
    reports_dir = Path(config.get("environmental_risk.outputs.reports_dir"))
    metrics_dir = Path(config.get("environmental_risk.outputs.metrics_dir"))
    plots_dir = Path(config.get("environmental_risk.outputs.plots_dir"))
    
    for d in [out_dir, reports_dir, metrics_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 1. Load Data
    env_df, epi_df = load_data(env_path, epi_path, region, disease)
    
    # 2. Align
    freq = config.get("environmental_risk.alignment.frequency", "D")
    aligned_df = align_datasets(env_df, epi_df, freq)
    
    # 3. Features
    variables_conf = config.get("environmental_risk.features.variables")
    lags = config.get("environmental_risk.features.lags")
    rolling = config.get("environmental_risk.features.rolling_windows")
    
    featured_df = build_features(aligned_df, variables_conf, lags, rolling)
    
    # 4. Lag Analysis
    target = config.get("environmental_risk.modeling.target")
    lag_corr = compute_lag_correlations(
        featured_df, target, list(variables_conf.keys()), lags, region, disease
    )
    lag_corr.to_csv(out_dir / "lag_correlations.csv", index=False)
    
    # 5. Statistical Model
    stat_preds = config.get("environmental_risk.modeling.statistical.predictors", [])
    stat_res = fit_negative_binomial_model(featured_df, target, stat_preds)
    
    # 6. ML Model
    ml_preds = config.get("environmental_risk.modeling.machine_learning.predictors", [])
    split_date = config.get("environmental_risk.modeling.split_date")
    ml_res = train_evaluate_ml_model(featured_df, target, ml_preds, split_date)
    
    # 7. Evaluate and Compare
    comparison_rows = []
    if ml_res['success']:
        y_test = ml_res['y_test']
        env_eval = evaluate_predictions(y_test, ml_res['predictions'])
        base_eval = evaluate_predictions(y_test, ml_res['baseline_preds'])
        
        comparison_rows.append({
            'region': region, 'disease': disease, 'model': 'Historical Mean (Baseline)',
            'training_period': f"<{split_date}", 'test_period': f">={split_date}",
            'MAE': base_eval['MAE'], 'RMSE': base_eval['RMSE'], 'R2': base_eval['R2'],
            'sample_count': ml_res['test_samples']
        })
        comparison_rows.append({
            'region': region, 'disease': disease, 'model': 'RandomForest (Env Features)',
            'training_period': f"<{split_date}", 'test_period': f">={split_date}",
            'MAE': env_eval['MAE'], 'RMSE': env_eval['RMSE'], 'R2': env_eval['R2'],
            'sample_count': ml_res['test_samples']
        })
        
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(metrics_dir / "environmental_model_comparison.csv", index=False)
        
    # 8. Risk Scoring
    weights = config.get("environmental_risk.scoring.weights")
    thresholds = config.get("environmental_risk.scoring.thresholds")
    scored_df = generate_risk_score(featured_df, weights, thresholds)
    
    score_out = scored_df[['date', 'region', 'disease', 'environmental_risk_score', 'risk_category', 'available_indicators', 'missing_indicators', 'score_method']].copy()
    score_out['data_provenance'] = "SOURCE_CALIBRATED_SYNTHETIC"
    score_out.to_csv(out_dir / "environmental_risk_scores.csv", index=False)
    
    # 9. Explanation
    explanation = generate_explanations(scored_df, region, disease)
    with open(reports_dir / "latest_explanation.txt", "w") as f:
        f.write(explanation)
        
    # 10. Plots
    plt.figure(figsize=(10,6))
    if ml_res['success']:
        plt.plot(ml_res['test_dates'], ml_res['y_test'], label='Observed')
        plt.plot(ml_res['test_dates'], ml_res['predictions'], label='Predicted (ML)')
        plt.legend()
        plt.title('Observed vs Predicted Cases')
        plt.savefig(plots_dir / "observed_vs_predicted.png")
        plt.close()
        
    # Report
    report = f"""# Environmental Risk Report
    
## Variables
{list(variables_conf.keys())}

## ML Comparison
{pd.DataFrame(comparison_rows).to_csv(index=False) if comparison_rows else 'N/A'}

## Latest Risk Explanation
{explanation}
"""
    with open(reports_dir / "environmental_risk_report.md", "w") as f:
        f.write(report)
        
    logger.info("Pipeline completed successfully.")
    
if __name__ == "__main__":
    run_pipeline()
