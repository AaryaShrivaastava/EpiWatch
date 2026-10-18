# Epidemiological Modeling Report

## Configuration
- Model Type: SIRD
- Disease: COVID-19
- Region: Bhopal
- Population: 2371061 (District Population 2011 Census)

## Parameters
- Beta (Transmission Rate): 1.0222
- Gamma (Recovery Rate): 1.0000
- Mu (Mortality Rate): 0.0113 (Applicable for SIRD)
- R0 (Basic Reproduction Number): 1.01

## Evaluation Metrics
{
  "MAE": 5.20241021911607,
  "RMSE": 6.407285788520782
}

## Limitations
- SIR/SIRD are simplified mathematical models.
- Reported cases are not identical to true infections.
- Underreporting and reporting delays may affect fitted parameters.
- Synthetic/source-calibrated observations cannot establish real-world epidemiological parameters.
- Simulation outputs are scenarios, not guaranteed forecasts.
- Results are for academic research and not medical or public-health decision-making.
