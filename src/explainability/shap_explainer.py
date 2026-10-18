"""SHAP tree explainer."""
import shap
import pandas as pd
from typing import Tuple, Any

def get_tree_explainer(model: Any, background_data: pd.DataFrame) -> shap.TreeExplainer:
    """Instantiate a SHAP TreeExplainer."""
    return shap.TreeExplainer(model, data=background_data, feature_perturbation="interventional")

def compute_shap_values(explainer: Any, X: pd.DataFrame) -> Tuple[Any, Any]:
    """Compute SHAP values for a dataset."""
    shap_values = explainer(X)
    
    # If the model is a classifier, shap_values may have shape (n_samples, n_features, n_classes)
    # We typically explain the positive class (index 1) if it's a binary classifier.
    if len(shap_values.shape) == 3:
        # shap_values[..., 1] selects the positive class for explanation
        return shap_values[..., 1], explainer.expected_value[1]
    
    # Otherwise regression or already single-output
    return shap_values, explainer.expected_value
