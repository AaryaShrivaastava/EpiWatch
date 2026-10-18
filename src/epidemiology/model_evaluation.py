"""Model evaluation for historical fits."""

import numpy as np
import pandas as pd
from typing import Dict

def evaluate_fit(
    observed: np.ndarray,
    simulated: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the fit between observed and simulated values.
    Uses MAE and RMSE.
    """
    if len(observed) != len(simulated):
        # Truncate to the shortest to compare overlapping period
        n = min(len(observed), len(simulated))
        observed = observed[:n]
        simulated = simulated[:n]
        
    mae = np.mean(np.abs(observed - simulated))
    rmse = np.sqrt(np.mean((observed - simulated)**2))
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse)
    }
