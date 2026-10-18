"""Scenario analysis module for epidemiological models."""
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt

from src.epidemiology.simulation import simulate_sir, simulate_sird
from src.epidemiology.reproduction_number import calculate_r0_sir, calculate_r0_sird

def run_scenarios_sir(
    scenarios: Dict[str, Dict[str, float]],
    N: float, S0: float, I0: float, R0: float,
    duration_days: int
) -> Dict[str, pd.DataFrame]:
    """
    Run multiple SIR scenarios.
    
    Args:
        scenarios: Dict mapping scenario name to parameters (beta, gamma).
    """
    results = {}
    for name, params in scenarios.items():
        df = simulate_sir(
            S0=S0, I0=I0, R0=R0, N=N,
            beta=params['beta'],
            gamma=params['gamma'],
            duration_days=duration_days
        )
        results[name] = df
    return results

def run_scenarios_sird(
    scenarios: Dict[str, Dict[str, float]],
    N: float, S0: float, I0: float, R0: float, D0: float,
    duration_days: int
) -> Dict[str, pd.DataFrame]:
    """
    Run multiple SIRD scenarios.
    """
    results = {}
    for name, params in scenarios.items():
        df = simulate_sird(
            S0=S0, I0=I0, R0=R0, D0=D0, N=N,
            beta=params['beta'],
            gamma=params['gamma'],
            mu=params.get('mu', 0.0),
            duration_days=duration_days
        )
        results[name] = df
    return results

def sensitivity_analysis_1d(
    base_params: Dict[str, float],
    param_name: str,
    param_values: List[float],
    model_type: str,
    N: float, S0: float, I0: float, R0: float, D0: float,
    duration_days: int
) -> pd.DataFrame:
    """
    Vary one parameter and measure peak infections and cumulative infections.
    """
    records = []
    
    for val in param_values:
        p = base_params.copy()
        p[param_name] = val
        
        if model_type.upper() == 'SIRD':
            df = simulate_sird(S0, I0, R0, D0, N, p['beta'], p['gamma'], p.get('mu', 0.0), duration_days)
            r0 = calculate_r0_sird(p['beta'], p['gamma'], p.get('mu', 0.0))
        else:
            df = simulate_sir(S0, I0, R0, N, p['beta'], p['gamma'], duration_days)
            r0 = calculate_r0_sir(p['beta'], p['gamma'])
            
        peak_infectious = df['I'].max()
        peak_day = df.loc[df['I'].idxmax(), 'day']
        cumulative_infections = df['new_infections'].sum()
        
        record = {
            'param_value': val,
            'peak_infectious': peak_infectious,
            'peak_day': peak_day,
            'cumulative_infections': cumulative_infections,
            'R0': r0
        }
        if model_type.upper() == 'SIRD':
            record['cumulative_deaths'] = df['new_deaths'].sum()
            
        records.append(record)
        
    return pd.DataFrame(records)
