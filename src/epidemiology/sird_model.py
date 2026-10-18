"""SIRD Model implementation."""
from typing import Tuple
import numpy as np

def sird_derivatives(y: Tuple[float, float, float, float], t: float, N: float, beta: float, gamma: float, mu: float) -> Tuple[float, float, float, float]:
    """
    Computes the derivatives for the SIRD model.
    
    Args:
        y: Tuple of (S, I, R, D) at time t
        t: Current time
        N: Total modeled population
        beta: Transmission rate
        gamma: Recovery rate
        mu: Disease-induced mortality rate
        
    Returns:
        Tuple of (dS/dt, dI/dt, dR/dt, dD/dt)
    """
    S, I, R, D = y
    
    # Check bounds
    if S < 0: S = 0
    if I < 0: I = 0
        
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    
    return dSdt, dIdt, dRdt, dDdt
