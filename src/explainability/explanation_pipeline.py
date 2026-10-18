"""Explanation Pipeline."""
import logging
import pandas as pd
import json
from pathlib import Path

from src.explainability.config import get_explain_config
from src.explainability.model_registry import load_model_and_preprocessor
from src.explainability.shap_explainer import get_tree_explainer, compute_shap_values
from src.explainability.global_explanations import calculate_global_importance, plot_shap_summary, plot_shap_bar
from src.explainability.local_explanations import generate_local_explanation

logger = logging.getLogger(__name__)

def run_explainability_pipeline():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Explainable AI (XAI) Engine (Phase 10)")
    
    config = get_explain_config()
    
    # Dirs
    global_dir = Path(config.get("explainability.outputs.global_dir", "reports/explainability/global"))
    local_dir = Path(config.get("explainability.outputs.local_dir", "data/explainability"))
    plots_dir = Path(config.get("explainability.outputs.plots_dir", "reports/explainability/plots"))
    
    for d in [global_dir, local_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Load dataset for explanation (we'll use epidemiological data as the evaluation set)
    epi_path = Path("data/features/epiwatch_master_features.csv")
    if not epi_path.exists():
        logger.error("No evaluation dataset found.")
        return
        
    df = pd.read_csv(epi_path)
    
    models_cfg = config.get("explainability.models", {})
    for model_key, m_cfg in models_cfg.items():
        logger.info(f"Explaining model: {model_key}")
        try:
            model, scaler, features = load_model_and_preprocessor(
                m_cfg['model_path'],
                m_cfg['scaler_path'],
                m_cfg['features_path']
            )
        except Exception as e:
            logger.error(f"Could not load model {model_key}: {e}")
            continue
            
        # Align features
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.error(f"Missing features {missing} in evaluation data.")
            continue
            
        X = df[features].copy()
        
        # Handle NA via simple fill since scaler doesn't handle NA natively in basic pipelines
        X = X.fillna(X.mean(numeric_only=True))
        
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features)
        
        # Sample for background
        bg_size = config.get("explainability.explanation_settings.background_samples", 100)
        eval_size = config.get("explainability.explanation_settings.max_global_samples", 500)
        
        if len(X_scaled_df) > bg_size:
            background = X_scaled_df.sample(n=bg_size, random_state=42)
        else:
            background = X_scaled_df
            
        if len(X_scaled_df) > eval_size:
            eval_data = X_scaled_df.sample(n=eval_size, random_state=42)
        else:
            eval_data = X_scaled_df
            
        # Tree explainer
        explainer = get_tree_explainer(model, background)
        shap_values, expected_value = compute_shap_values(explainer, eval_data)
        
        # Global Importance
        importance_df = calculate_global_importance(shap_values, features)
        importance_df.to_csv(global_dir / f"{model_key}_global_importance.csv", index=False)
        
        # Plots
        try:
            plot_shap_summary(shap_values, plots_dir / f"{model_key}_summary_plot.png")
            plot_shap_bar(shap_values, plots_dir / f"{model_key}_bar_plot.png")
        except Exception as e:
            logger.error(f"Failed to plot SHAP for {model_key}: {e}")
            
        # Local Explanation (just take the first observation from eval_data as an example)
        idx = eval_data.index[0]
        actual_pred = model.predict(eval_data.iloc[[0]])[0]
        
        # shap_values.values[0] accesses the SHAP values array for the first instance
        local_exp = generate_local_explanation(
            identifier=f"row_{idx}",
            shap_vals=shap_values.values[0],
            feature_vals=eval_data.iloc[0].values,
            feature_names=features,
            expected_value=expected_value,
            actual_pred=actual_pred
        )
        
        with open(local_dir / f"{model_key}_local_example.json", "w") as f:
            json.dump(local_exp, f, indent=2)
            
        logger.info(f"Successfully generated explanations for {model_key}")
        
    logger.info("Explainability Pipeline completed.")

if __name__ == "__main__":
    run_explainability_pipeline()
