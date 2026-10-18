"""SIR Model implementation."""
from typing import Tuple
import numpy as np

def sir_derivatives(y: Tuple[float, float, float], t: float, N: float, beta: float, gamma: float) -> Tuple[float, float, float]:
    """
    Computes the derivatives for the SIR model.
    
    Args:
        y: Tuple of (S, I, R) at time t
        t: Current time (not used directly since model is autonomous, but required by solvers)
        N: Total modeled population
        beta: Transmission rate
        gamma: Recovery rate
        
    Returns:
        Tuple of (dS/dt, dI/dt, dR/dt)
    """
    S, I, R = y
    
    # Check bounds (numerical safety)
    if S < 0: S = 0
    if I < 0: I = 0
        
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    
    return dSdt, dIdt, dRdt
