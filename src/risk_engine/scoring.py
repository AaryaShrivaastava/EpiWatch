"""Integrated scoring and classification."""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def calculate_integrated_score(assessments: Dict[str, Dict[str, Any]], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Calculate the weighted integrated risk score dynamically ignoring missing."""
    
    total_weight = 0.0
    weighted_sum = 0.0
    contributions = {}
    
    for key, assessment in assessments.items():
        if key in weights and assessment['status'] == 'available':
            w = weights[key]
            val = assessment['score']
            weighted_sum += val * w
            total_weight += w
            contributions[key] = val * w
            
    if total_weight > 0:
        final_score = weighted_sum / total_weight
        # Normalize contributions
        contributions = {k: v / total_weight for k, v in contributions.items()}
    else:
        final_score = np.nan
        
    return final_score, contributions

def classify_risk(score: float, thresholds: dict) -> str:
    """Classify 0-100 score."""
    if pd.isna(score):
        return "Insufficient Data"
    if score < thresholds.get('moderate', 25): return "Low"
    if score < thresholds.get('high', 50): return "Moderate"
    if score < thresholds.get('very_high', 75): return "High"
    return "Very High"
