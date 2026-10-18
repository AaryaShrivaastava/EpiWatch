"""
Population modeling utilities.
"""
from typing import Dict, Tuple

def initialize_compartments(
    total_population: float,
    initial_infectious: float,
    initial_recovered: float,
    initial_deaths: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Initializes compartments for SIR/SIRD models.
    
    Returns:
        Tuple of (S, I, R, D)
    """
    if total_population <= 0:
        raise ValueError("Total population must be positive.")
    if initial_infectious < 0 or initial_recovered < 0 or initial_deaths < 0:
        raise ValueError("Compartment counts must be non-negative.")
        
    non_susceptible = initial_infectious + initial_recovered + initial_deaths
    if non_susceptible > total_population:
        raise ValueError("Initial non-susceptible population cannot exceed total modeled population.")
        
    initial_susceptible = total_population - non_susceptible
    
    return initial_susceptible, initial_infectious, initial_recovered, initial_deaths
