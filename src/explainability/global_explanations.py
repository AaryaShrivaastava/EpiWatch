"""Global explanations and summary metrics."""
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

def calculate_global_importance(shap_values: Any, feature_names: List[str]) -> pd.DataFrame:
    """Calculate mean absolute SHAP importance."""
    # shap_values.values shape is (n_samples, n_features)
    vals = np.abs(shap_values.values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': vals
    }).sort_values('mean_abs_shap', ascending=False)
    
    return importance_df

def plot_shap_summary(shap_values: Any, out_path: Path):
    """Generate SHAP summary (beeswarm) plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
def plot_shap_bar(shap_values: Any, out_path: Path):
    """Generate SHAP bar plot."""
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
