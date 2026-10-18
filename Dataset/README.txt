EpiWatch — Synthetic Bhopal Dataset Pack
==============================================

Purpose
-------
Synthetic data for developing/testing the EpiWatch epidemiological intelligence platform.

Geographic anchor
-----------------
Region: Bhopal district, Madhya Pradesh, India
Population anchor: 2,368,145 (Census 2011 / Government of Madhya Pradesh district portal)
Area: 2,772 sq km
Population density: 855 persons/sq km
Male population: 1,239,378
Female population: 1,128,767
Sex ratio: 918 females per 1000 males
Literacy: 80.37%

Important
---------
ONLY the demographic anchor values above are sourced from official Census/Government pages.
The disease cases, healthcare capacity, weather/environmental observations, interventions,
transmission parameters and derived risk labels in these CSVs are SYNTHETIC and must not be
represented as actual Bhopal public-health measurements.

Files
-----
01_demographics_bhopal.csv
    Static demographic/geographic features.

02_environmental_daily_bhopal.csv
    Daily synthetic temperature, rainfall, humidity, water accumulation, AQI, wind,
    solar radiation, surface-water and pollution indicators (2021-2025).

03_healthcare_monthly_bhopal.csv
    Monthly synthetic hospital/ICU/testing/oxygen/medicine/emergency-capacity indicators.

04_interventions_daily_bhopal.csv
    Synthetic intervention/testing/public-awareness/vector-control indicators.

05_epidemiological_daily_bhopal.csv
    Daily synthetic disease observations for Dengue, Malaria, Influenza, COVID-19 and Typhoid.

06_disease_transmission_parameters.csv
    Synthetic modelling parameters for SIR/SIRD experiments.

07_epiwatch_derived_features_bhopal.csv
    Derived rolling averages, growth rate, incidence, and synthetic risk labels for pipeline testing.

Recommended modelling order
---------------------------
1. Outbreak detection: 05 + 02
2. Forecasting: 05
3. Environmental correlation: 05 + 02
4. Healthcare stress/preparedness: 05 + 03
5. Intervention analysis: 05 + 04
6. Risk scoring: combine 01 + 02 + 03 + 05
7. SIR/SIRD experiments: 05 + 06

Do not use the synthetic values for real medical, public-health, policy, or clinical decisions.
