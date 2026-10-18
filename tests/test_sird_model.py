"""Tests for SIRD model."""
import pytest
import numpy as np
from src.epidemiology.sird_model import sird_derivatives

def test_sird_derivatives_conservation():
    S, I, R, D = 980, 10, 5, 5
    N = 1000
    beta, gamma, mu = 0.3, 0.1, 0.05
    dSdt, dIdt, dRdt, dDdt = sird_derivatives((S, I, R, D), 0, N, beta, gamma, mu)
    
    assert np.isclose(dSdt + dIdt + dRdt + dDdt, 0.0)

def test_sird_derivatives_values():
    S, I, R, D = 100, 10, 0, 0
    N = 110
    beta = 0.5
    gamma = 0.1
    mu = 0.02
    
    expected_dSdt = -0.5 * 100 * 10 / 110
    expected_dRdt = 0.1 * 10
    expected_dDdt = 0.02 * 10
    expected_dIdt = -expected_dSdt - expected_dRdt - expected_dDdt
    
    dSdt, dIdt, dRdt, dDdt = sird_derivatives((S, I, R, D), 0, N, beta, gamma, mu)
    
    assert np.isclose(dSdt, expected_dSdt)
    assert np.isclose(dIdt, expected_dIdt)
    assert np.isclose(dRdt, expected_dRdt)
    assert np.isclose(dDdt, expected_dDdt)
