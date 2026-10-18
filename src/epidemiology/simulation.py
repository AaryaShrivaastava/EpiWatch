"""Simulation module for epidemiology models."""

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from typing import Tuple, Optional

from src.epidemiology.sir_model import sir_derivatives
from src.epidemiology.sird_model import sird_derivatives

def simulate_sir(
    S0: float, I0: float, R0: float,
    N: float, beta: float, gamma: float,
    duration_days: int, time_step: float = 1.0
) -> pd.DataFrame:
    """
    Simulate the SIR model over a given duration.
    """
    t = np.arange(0, duration_days + time_step, time_step)
    y0 = (S0, I0, R0)
    
    result = odeint(sir_derivatives, y0, t, args=(N, beta, gamma))
    S, I, R = result.T
    
    # Calculate incidence (new infections)
    # new_infections = beta * S * I / N * time_step
    # Note: ODE results might have small numerical issues, we ensure non-negativity
    S = np.maximum(S, 0)
    I = np.maximum(I, 0)
    R = np.maximum(R, 0)
    
    new_infections = beta * S * I / N * time_step
    new_recoveries = gamma * I * time_step
    
    df = pd.DataFrame({
        'day': t,
        'S': S,
        'I': I,
        'R': R,
        'new_infections': new_infections,
        'new_recoveries': new_recoveries
    })
    
    # Ensure conservation
    df['total'] = df['S'] + df['I'] + df['R']
    
    return df

def simulate_sird(
    S0: float, I0: float, R0: float, D0: float,
    N: float, beta: float, gamma: float, mu: float,
    duration_days: int, time_step: float = 1.0
) -> pd.DataFrame:
    """
    Simulate the SIRD model over a given duration.
    """
    t = np.arange(0, duration_days + time_step, time_step)
    y0 = (S0, I0, R0, D0)
    
    result = odeint(sird_derivatives, y0, t, args=(N, beta, gamma, mu))
    S, I, R, D = result.T
    
    S = np.maximum(S, 0)
    I = np.maximum(I, 0)
    R = np.maximum(R, 0)
    D = np.maximum(D, 0)
    
    new_infections = beta * S * I / N * time_step
    new_recoveries = gamma * I * time_step
    new_deaths = mu * I * time_step
    
    df = pd.DataFrame({
        'day': t,
        'S': S,
        'I': I,
        'R': R,
        'D': D,
        'new_infections': new_infections,
        'new_recoveries': new_recoveries,
        'new_deaths': new_deaths
    })
    
    df['total'] = df['S'] + df['I'] + df['R'] + df['D']
    return df
