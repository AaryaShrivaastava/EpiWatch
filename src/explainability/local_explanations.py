"""Local explanation generation."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List

def generate_local_explanation(
    identifier: str,
    shap_vals: np.ndarray,
    feature_vals: np.ndarray,
    feature_names: List[str],
    expected_value: float,
    actual_pred: float
) -> Dict[str, Any]:
    """Generate local explanation for a single observation."""
    
    # Pair features with their SHAP values
    contributions = []
    for name, s_val, f_val in zip(feature_names, shap_vals, feature_vals):
        contributions.append({
            'feature': name,
            'feature_value': float(f_val),
            'shap_value': float(s_val),
            'direction': 'up' if s_val > 0 else 'down' if s_val < 0 else 'neutral'
        })
        
    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    # Textual summary
    top_up = [c['feature'] for c in contributions if c['direction'] == 'up'][:2]
    top_down = [c['feature'] for c in contributions if c['direction'] == 'down'][:2]
    
    text = "The model prediction was influenced "
    if top_up:
        text += f"upward by {', '.join(top_up).replace('_', ' ')}. "
    if top_down:
        text += f"Seasonal or opposing features like {', '.join(top_down).replace('_', ' ')} contributed in the opposite direction. "
        
    text += "These are model-attributed influences, not proof of causal relationships."
    
    return {
        'observation_id': identifier,
        'expected_value': float(expected_value),
        'actual_prediction': float(actual_pred),
        'top_contributions': contributions[:5], # Keep top 5
        'explanation_text': text
    }
