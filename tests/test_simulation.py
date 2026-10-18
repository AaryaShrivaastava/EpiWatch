"""Tests for Simulation Module."""
import pytest
import numpy as np
from src.epidemiology.simulation import simulate_sir, simulate_sird

def test_simulate_sir():
    S0, I0, R0 = 990, 10, 0
    N = 1000
    beta, gamma = 0.3, 0.1
    duration = 10
    
    df = simulate_sir(S0, I0, R0, N, beta, gamma, duration)
    
    assert len(df) == duration + 1
    assert 'S' in df.columns
    assert 'I' in df.columns
    assert 'R' in df.columns
    
    # Check total population conservation
    assert np.allclose(df['total'], N)
    
    # Check non-negativity
    assert (df['S'] >= 0).all()
    assert (df['I'] >= 0).all()
    assert (df['R'] >= 0).all()

def test_simulate_sird():
    S0, I0, R0, D0 = 990, 10, 0, 0
    N = 1000
    beta, gamma, mu = 0.3, 0.1, 0.01
    duration = 10
    
    df = simulate_sird(S0, I0, R0, D0, N, beta, gamma, mu, duration)
    
    assert len(df) == duration + 1
    assert 'D' in df.columns
    assert np.allclose(df['total'], N)
    assert (df['D'] >= 0).all()
