"""Tests for SIR model."""
import pytest
import numpy as np
from src.epidemiology.sir_model import sir_derivatives

def test_sir_derivatives_conservation():
    S, I, R = 990, 10, 0
    N = 1000
    beta, gamma = 0.3, 0.1
    dSdt, dIdt, dRdt = sir_derivatives((S, I, R), 0, N, beta, gamma)
    
    # Conservation of mass: dS/dt + dI/dt + dR/dt = 0
    assert np.isclose(dSdt + dIdt + dRdt, 0.0)

def test_sir_derivatives_values():
    S, I, R = 100, 10, 0
    N = 110
    beta = 0.5
    gamma = 0.1
    
    expected_dSdt = -0.5 * 100 * 10 / 110
    expected_dRdt = 0.1 * 10
    expected_dIdt = -expected_dSdt - expected_dRdt
    
    dSdt, dIdt, dRdt = sir_derivatives((S, I, R), 0, N, beta, gamma)
    
    assert np.isclose(dSdt, expected_dSdt)
    assert np.isclose(dIdt, expected_dIdt)
    assert np.isclose(dRdt, expected_dRdt)

def test_sir_derivatives_non_negative():
    S, I, R = -10, -5, 0
    N = 100
    beta, gamma = 0.3, 0.1
    dSdt, dIdt, dRdt = sir_derivatives((S, I, R), 0, N, beta, gamma)
    
    # Negative inputs should be floored to 0 inside derivatives for safety
    # dSdt = -beta * 0 * 0 = 0
    assert np.isclose(dSdt, 0.0)
    assert np.isclose(dIdt, 0.0)
    assert np.isclose(dRdt, 0.0)
