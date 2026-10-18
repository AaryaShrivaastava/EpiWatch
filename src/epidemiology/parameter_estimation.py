"""Parameter estimation using scipy.optimize."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, Any, Optional

from src.epidemiology.simulation import simulate_sir, simulate_sird
from src.epidemiology.population import initialize_compartments

def sir_loss_function(params: Tuple[float, float], *args) -> float:
    beta, gamma = params
    S0, I0, R0, N, observed_cases, time_step = args
    
    # Simulate
    duration = (len(observed_cases) - 1) * time_step
    sim_df = simulate_sir(S0, I0, R0, N, beta, gamma, duration, time_step)
    
    # We compare simulated new infections with observed daily incidence
    # Caution: observed cases are often underreported. Here we assume 
    # observed cases proxy true incidence.
    simulated_incidence = sim_df['new_infections'].values
    
    # Mean Squared Error
    if len(simulated_incidence) != len(observed_cases):
        return float('inf')
        
    mse = np.mean((simulated_incidence - observed_cases) ** 2)
    return float(mse)

def sird_loss_function(params: Tuple[float, float, float], *args) -> float:
    beta, gamma, mu = params
    S0, I0, R0, D0, N, observed_cases, observed_deaths, time_step = args
    
    duration = (len(observed_cases) - 1) * time_step
    sim_df = simulate_sird(S0, I0, R0, D0, N, beta, gamma, mu, duration, time_step)
    
    sim_infections = sim_df['new_infections'].values
    sim_deaths = sim_df['new_deaths'].values
    
    if len(sim_infections) != len(observed_cases):
        return float('inf')
        
    mse_infections = np.mean((sim_infections - observed_cases) ** 2)
    mse_deaths = np.mean((sim_deaths - observed_deaths) ** 2) if observed_deaths is not None else 0
    
    # Weight the losses
    return float(mse_infections + mse_deaths * 10) # arbitrary weighting


def estimate_parameters_sir(
    observed_cases: np.ndarray,
    total_population: float,
    initial_infectious: float,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.01, 2.0), (0.01, 1.0)),
    time_step: float = 1.0
) -> Dict[str, Any]:
    """Estimate beta and gamma for SIR."""
    S0, I0, R0, D0 = initialize_compartments(total_population, initial_infectious, 0)
    
    initial_guess = [0.3, 0.1]
    args = (S0, I0, R0, total_population, observed_cases, time_step)
    
    result = minimize(
        sir_loss_function,
        initial_guess,
        args=args,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    return {
        'success': result.success,
        'beta': float(result.x[0]),
        'gamma': float(result.x[1]),
        'loss': float(result.fun),
        'message': result.message
    }

def estimate_parameters_sird(
    observed_cases: np.ndarray,
    observed_deaths: Optional[np.ndarray],
    total_population: float,
    initial_infectious: float,
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((0.01, 2.0), (0.01, 1.0), (0.0, 0.1)),
    time_step: float = 1.0
) -> Dict[str, Any]:
    """Estimate beta, gamma, and mu for SIRD."""
    S0, I0, R0, D0 = initialize_compartments(total_population, initial_infectious, 0, 0)
    
    initial_guess = [0.3, 0.1, 0.01]
    args = (S0, I0, R0, D0, total_population, observed_cases, observed_deaths, time_step)
    
    result = minimize(
        sird_loss_function,
        initial_guess,
        args=args,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    return {
        'success': result.success,
        'beta': float(result.x[0]),
        'gamma': float(result.x[1]),
        'mu': float(result.x[2]),
        'loss': float(result.fun),
        'message': result.message
    }
