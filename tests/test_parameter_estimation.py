"""Tests for Parameter Estimation."""
import pytest
import numpy as np
from src.epidemiology.parameter_estimation import estimate_parameters_sir

def test_estimate_parameters_sir():
    # Create synthetic observed data that perfectly matches beta=0.3, gamma=0.1
    from src.epidemiology.simulation import simulate_sir
    N = 1000
    S0, I0, R0 = 990, 10, 0
    true_beta = 0.3
    true_gamma = 0.1
    duration = 20
    
    sim_df = simulate_sir(S0, I0, R0, N, true_beta, true_gamma, duration)
    observed_cases = sim_df['new_infections'].values
    
    # Now try to estimate parameters from this synthetic data
    result = estimate_parameters_sir(
        observed_cases, N, I0,
        bounds=((0.1, 0.5), (0.05, 0.2))
    )
    
    # Optimization can sometimes terminate with warning but correct values
    # We mainly care about the parameter estimates being close
    assert np.isclose(result['beta'], true_beta, atol=0.05)
    assert np.isclose(result['gamma'], true_gamma, atol=0.05)
