"""Epidemiology Pipeline."""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path

from src.epidemiology.config import get_epi_config
from src.epidemiology.population import initialize_compartments
from src.epidemiology.reproduction_number import calculate_r0_sir, calculate_r0_sird
from src.epidemiology.simulation import simulate_sir, simulate_sird
from src.epidemiology.parameter_estimation import estimate_parameters_sir, estimate_parameters_sird
from src.epidemiology.scenario_analysis import run_scenarios_sir, run_scenarios_sird, sensitivity_analysis_1d
from src.epidemiology.model_evaluation import evaluate_fit
from src.epidemiology.visualization import plot_compartments, plot_incidence, plot_scenarios, plot_sensitivity

logger = logging.getLogger(__name__)

def run_pipeline():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting EpiWatch Phase 5 Pipeline")
    
    config = get_epi_config()
    
    # 1. Inspect and load data
    data_path = Path("data/processed/epidemiological_clean.csv")
    observed_cases = None
    observed_deaths = None
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        disease = config.get("epidemiology.disease", "COVID-19")
        region = config.get("epidemiology.region", "Bhopal")
        
        subset = df[(df['disease'] == disease) & (df['region'] == region)].copy()
        if not subset.empty:
            subset['date'] = pd.to_datetime(subset['date'])
            subset = subset.sort_values('date')
            
            # Filter by period if configured
            start_date = config.get("epidemiology.fitting.period_start")
            end_date = config.get("epidemiology.fitting.period_end")
            
            if start_date and end_date:
                mask = (subset['date'] >= start_date) & (subset['date'] <= end_date)
                subset = subset[mask]
                
            observed_cases = subset['new_cases'].values
            if 'deaths' in subset.columns: # Assuming daily new deaths is recorded as 'deaths' or we need to diff it
                # if 'deaths' is cumulative, we would diff. Let's assume daily new_deaths exists or 'deaths' is daily
                if 'new_deaths' in subset.columns:
                    observed_deaths = subset['new_deaths'].values
                else:
                    observed_deaths = subset['deaths'].values # Might be cumulative in some datasets, check first
    
    # 2. Population and assumptions
    pop_val = config.get("epidemiology.population.value", 1000000)
    I0 = config.get("epidemiology.initial_compartments.infectious", 10)
    R0 = config.get("epidemiology.initial_compartments.recovered", 0)
    D0 = config.get("epidemiology.initial_compartments.deaths", 0)
    
    S0, I0, R0, D0 = initialize_compartments(pop_val, I0, R0, D0)
    
    model_type = config.get("epidemiology.model_type", "SIR")
    duration = config.get("epidemiology.simulation.duration_days", 100)
    
    beta = config.get("epidemiology.parameters.beta", 0.3)
    gamma = config.get("epidemiology.parameters.gamma", 0.1)
    mu = config.get("epidemiology.parameters.mu", 0.01)
    
    out_dir = Path(config.get("epidemiology.outputs.data_dir", "data/epidemiology"))
    reports_dir = Path(config.get("epidemiology.outputs.reports_dir", "reports/epidemiology"))
    plots_dir = Path(config.get("epidemiology.outputs.plots_dir", "reports/epidemiology/plots"))
    
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fit_metrics = {}
    r0_val = None
    
    # 3. Parameter Estimation
    if observed_cases is not None and len(observed_cases) > 10:
        logger.info(f"Estimating parameters using {len(observed_cases)} observations.")
        if model_type == "SIR":
            est = estimate_parameters_sir(observed_cases, pop_val, I0)
            if est['success']:
                beta = est['beta']
                gamma = est['gamma']
                logger.info(f"Estimated parameters: beta={beta:.4f}, gamma={gamma:.4f}")
        elif model_type == "SIRD":
            est = estimate_parameters_sird(observed_cases, observed_deaths, pop_val, I0)
            if est['success']:
                beta = est['beta']
                gamma = est['gamma']
                mu = est['mu']
                logger.info(f"Estimated parameters: beta={beta:.4f}, gamma={gamma:.4f}, mu={mu:.4f}")

    # 4. Calculate R0
    if model_type == "SIR":
        r0_val = calculate_r0_sir(beta, gamma)
    else:
        r0_val = calculate_r0_sird(beta, gamma, mu)
    logger.info(f"Calculated R0: {r0_val:.2f}")
    
    # 5. Simulate
    logger.info(f"Running {model_type} simulation...")
    if model_type == "SIR":
        sim_df = simulate_sir(S0, I0, R0, pop_val, beta, gamma, duration)
        sim_df.to_csv(out_dir / "sir_simulation.csv", index=False)
    else:
        sim_df = simulate_sird(S0, I0, R0, D0, pop_val, beta, gamma, mu, duration)
        sim_df.to_csv(out_dir / "sird_simulation.csv", index=False)
        
    # 6. Evaluate Fit
    if observed_cases is not None:
        fit_metrics = evaluate_fit(observed_cases, sim_df['new_infections'].values)
        logger.info(f"Fit metrics: {fit_metrics}")
        
    # 7. Scenario Analysis
    scenarios_config = config.get("epidemiology.scenarios", {})
    if scenarios_config:
        if model_type == "SIR":
            scenario_res = run_scenarios_sir(scenarios_config, pop_val, S0, I0, R0, duration)
        else:
            scenario_res = run_scenarios_sird(scenarios_config, pop_val, S0, I0, R0, D0, duration)
            
        for name, sdf in scenario_res.items():
            sdf.to_csv(out_dir / f"scenario_{name}.csv", index=False)
            
        plot_scenarios(scenario_res, plots_dir / "scenario_comparison.png")
        
    # 8. Sensitivity Analysis
    sens_df = sensitivity_analysis_1d(
        {'beta': beta, 'gamma': gamma, 'mu': mu},
        'beta',
        [0.1, 0.2, 0.3, 0.4, 0.5],
        model_type, pop_val, S0, I0, R0, D0, duration
    )
    sens_df.to_csv(out_dir / "sensitivity_beta.csv", index=False)
    plot_sensitivity(sens_df, 'beta', plots_dir / "sensitivity_beta.png")
    
    # 9. Plotting
    plot_compartments(sim_df, f"{model_type} Simulation", plots_dir / f"{model_type.lower()}_compartments.png")
    obs_series = pd.Series(observed_cases) if observed_cases is not None else None
    plot_incidence(sim_df, obs_series, f"{model_type} Simulated vs Observed Incidence", plots_dir / "incidence_comparison.png")
    
    # 10. Write report
    report_content = f"""# Epidemiological Modeling Report

## Configuration
- Model Type: {model_type}
- Disease: {config.get('epidemiology.disease')}
- Region: {config.get('epidemiology.region')}
- Population: {pop_val} ({config.get('epidemiology.population.assumption')})

## Parameters
- Beta (Transmission Rate): {beta:.4f}
- Gamma (Recovery Rate): {gamma:.4f}
- Mu (Mortality Rate): {mu:.4f} (Applicable for SIRD)
- R0 (Basic Reproduction Number): {r0_val:.2f}

## Evaluation Metrics
{json.dumps(fit_metrics, indent=2)}

## Limitations
- SIR/SIRD are simplified mathematical models.
- Reported cases are not identical to true infections.
- Underreporting and reporting delays may affect fitted parameters.
- Synthetic/source-calibrated observations cannot establish real-world epidemiological parameters.
- Simulation outputs are scenarios, not guaranteed forecasts.
- Results are for academic research and not medical or public-health decision-making.
"""
    with open(reports_dir / "epidemiology_model_report.md", "w") as f:
        f.write(report_content)
        
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
