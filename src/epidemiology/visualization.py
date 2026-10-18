"""Epidemiology visualization module."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_compartments(df: pd.DataFrame, title: str, output_path: Path):
    """Plot S, I, R (and D if present) over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['day'], df['S'], label='Susceptible (S)', color='blue')
    plt.plot(df['day'], df['I'], label='Infectious (I)', color='red')
    plt.plot(df['day'], df['R'], label='Recovered (R)', color='green')
    if 'D' in df.columns:
        plt.plot(df['day'], df['D'], label='Deaths (D)', color='black')
        
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_incidence(sim_df: pd.DataFrame, observed_cases: pd.Series, title: str, output_path: Path):
    """Compare simulated new infections against observed cases."""
    plt.figure(figsize=(10, 6))
    plt.plot(sim_df['day'], sim_df['new_infections'], label='Simulated Incidence', color='orange')
    
    if observed_cases is not None and not observed_cases.empty:
        days = range(len(observed_cases))
        plt.scatter(days, observed_cases, label='Observed Cases', color='red', alpha=0.5, s=15)
        
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("New Cases / Infections")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_scenarios(scenarios: dict, output_path: Path):
    """Plot infectious curves for multiple scenarios."""
    plt.figure(figsize=(10, 6))
    for name, df in scenarios.items():
        plt.plot(df['day'], df['I'], label=f"{name} - Infectious")
        
    plt.title("Scenario Comparison - Infectious Population")
    plt.xlabel("Day")
    plt.ylabel("Infectious Population")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_sensitivity(sensitivity_df: pd.DataFrame, param_name: str, output_path: Path):
    """Plot sensitivity of peak infectious to a parameter."""
    plt.figure(figsize=(8, 5))
    plt.plot(sensitivity_df['param_value'], sensitivity_df['peak_infectious'], marker='o')
    plt.title(f"Sensitivity Analysis: Peak Infectious vs {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Peak Infectious Population")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
