"""EpiWatch Multi-Region Demonstration Data Generator.

Generates realistic, scientifically constrained synthetic epidemiological,
environmental, demographic, and healthcare datasets for 5 key regions
in Madhya Pradesh, India: Bhopal, Indore, Jabalpur, Gwalior, and Ujjain.

Key Features:
- Preserves official Census 2011 demographic anchors (OFFICIAL_STATIC_REFERENCE).
- Generates disease-specific seasonal and lagged dynamics for 5 diseases:
  Dengue, Malaria, COVID-19, Influenza, and Typhoid.
- Implements realistic biological constraints: non-negativity, valid test positivity,
  bed occupancy, hospital admissions, and controlled surge events for outbreak detection.
- Emits explicit data provenance metadata (SYNTHETIC_DEMO).
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Official Census 2011 anchors for Madhya Pradesh districts
DEMOGRAPHIC_ANCHORS = {
    "Bhopal": {
        "region": "Bhopal",
        "state": "Madhya Pradesh",
        "country": "India",
        "area_sq_km": 2772,
        "population_2011": 2368145,
        "population_density_per_sq_km": 855,
        "urban_population_2011": 1917051,
        "rural_population_2011": 454010,
        "male_population_2011": 1239378,
        "female_population_2011": 1128767,
        "sex_ratio_females_per_1000_males": 918,
        "literacy_rate_percent": 80.37,
        "age_0_6_percent": 12.2,
        "age_7_14_percent": 15.4,
        "age_15_24_percent": 20.3,
        "age_25_44_percent": 30.1,
        "age_45_59_percent": 13.0,
        "age_60_plus_percent": 9.0,
        "urbanization_percent": 80.85,
        "avg_household_size": 4.4,
        "housing_crowding_index": 0.56,
        "population_growth_index_annual": 1.8,
        "data_status": "OFFICIAL_STATIC_REFERENCE"
    },
    "Indore": {
        "region": "Indore",
        "state": "Madhya Pradesh",
        "country": "India",
        "area_sq_km": 3898,
        "population_2011": 3276697,
        "population_density_per_sq_km": 841,
        "urban_population_2011": 2427709,
        "rural_population_2011": 848988,
        "male_population_2011": 1699627,
        "female_population_2011": 1577070,
        "sex_ratio_females_per_1000_males": 928,
        "literacy_rate_percent": 80.87,
        "age_0_6_percent": 12.5,
        "age_7_14_percent": 15.1,
        "age_15_24_percent": 21.0,
        "age_25_44_percent": 31.0,
        "age_45_59_percent": 12.4,
        "age_60_plus_percent": 8.0,
        "urbanization_percent": 74.09,
        "avg_household_size": 4.6,
        "housing_crowding_index": 0.62,
        "population_growth_index_annual": 2.1,
        "data_status": "OFFICIAL_STATIC_REFERENCE"
    },
    "Jabalpur": {
        "region": "Jabalpur",
        "state": "Madhya Pradesh",
        "country": "India",
        "area_sq_km": 5210,
        "population_2011": 2463289,
        "population_density_per_sq_km": 473,
        "urban_population_2011": 1440120,
        "rural_population_2011": 1023169,
        "male_population_2011": 1276940,
        "female_population_2011": 1186349,
        "sex_ratio_females_per_1000_males": 929,
        "literacy_rate_percent": 81.07,
        "age_0_6_percent": 12.1,
        "age_7_14_percent": 15.8,
        "age_15_24_percent": 19.8,
        "age_25_44_percent": 29.5,
        "age_45_59_percent": 13.5,
        "age_60_plus_percent": 9.3,
        "urbanization_percent": 58.46,
        "avg_household_size": 4.5,
        "housing_crowding_index": 0.51,
        "population_growth_index_annual": 1.45,
        "data_status": "OFFICIAL_STATIC_REFERENCE"
    },
    "Gwalior": {
        "region": "Gwalior",
        "state": "Madhya Pradesh",
        "country": "India",
        "area_sq_km": 4560,
        "population_2011": 2032036,
        "population_density_per_sq_km": 446,
        "urban_population_2011": 1273792,
        "rural_population_2011": 758244,
        "male_population_2011": 1090145,
        "female_population_2011": 941891,
        "sex_ratio_females_per_1000_males": 864,
        "literacy_rate_percent": 76.65,
        "age_0_6_percent": 13.4,
        "age_7_14_percent": 16.2,
        "age_15_24_percent": 20.1,
        "age_25_44_percent": 28.8,
        "age_45_59_percent": 12.8,
        "age_60_plus_percent": 8.7,
        "urbanization_percent": 62.69,
        "avg_household_size": 4.8,
        "housing_crowding_index": 0.58,
        "population_growth_index_annual": 1.6,
        "data_status": "OFFICIAL_STATIC_REFERENCE"
    },
    "Ujjain": {
        "region": "Ujjain",
        "state": "Madhya Pradesh",
        "country": "India",
        "area_sq_km": 6091,
        "population_2011": 1986864,
        "population_density_per_sq_km": 326,
        "urban_population_2011": 778715,
        "rural_population_2011": 1208149,
        "male_population_2011": 1015863,
        "female_population_2011": 971001,
        "sex_ratio_females_per_1000_males": 956,
        "literacy_rate_percent": 72.34,
        "age_0_6_percent": 13.0,
        "age_7_14_percent": 16.0,
        "age_15_24_percent": 19.5,
        "age_25_44_percent": 29.0,
        "age_45_59_percent": 13.2,
        "age_60_plus_percent": 9.3,
        "urbanization_percent": 39.19,
        "avg_household_size": 4.7,
        "housing_crowding_index": 0.49,
        "population_growth_index_annual": 1.5,
        "data_status": "OFFICIAL_STATIC_REFERENCE"
    }
}

DISEASES = ["Dengue", "Malaria", "COVID-19", "Influenza", "Typhoid"]


def generate_environmental_series(dates: pd.DatetimeIndex, region: str, rng: np.random.Generator) -> pd.DataFrame:
    """Generate physically plausible daily weather series for a central Indian district."""
    n_days = len(dates)
    doy = dates.dayofyear.values

    # Base temperatures (seasonal sine wave with summer peak around May/June)
    base_temp = 26.0 - 9.0 * np.cos(2 * np.pi * (doy - 25) / 365.25)
    temp_noise = rng.normal(0, 1.8, n_days)
    temperature_c = np.round(np.clip(base_temp + temp_noise, 8.0, 46.0), 2)

    # Monsoon rainfall (June through September: DOY ~150 to ~270)
    monsoon_mask = (doy >= 160) & (doy <= 270)
    rain_prob = np.where(monsoon_mask, 0.55, 0.05)
    has_rain = rng.binomial(1, rain_prob, n_days)
    rain_amount = rng.exponential(scale=16.0, size=n_days) * has_rain
    rainfall_mm = np.round(rain_amount, 2)

    # Relative humidity (high during monsoon, low in hot summer)
    base_humidity = 55.0 - 25.0 * np.sin(2 * np.pi * (doy - 40) / 365.25)
    base_humidity = np.where(monsoon_mask, base_humidity + 25.0, base_humidity)
    humidity_percent = np.round(np.clip(base_humidity + rng.normal(0, 5.0, n_days), 15.0, 95.0), 2)

    # Wind speed
    wind_speed_kmh = np.round(np.clip(rng.normal(9.0, 3.2, n_days), 2.0, 38.0), 2)

    # Solar radiation (kWh/m2)
    solar_radiation = np.round(np.clip(5.5 - 1.5 * np.where(monsoon_mask, 1.0, 0.0) + rng.normal(0, 0.6, n_days), 1.5, 7.8), 2)

    # Air Quality Index (AQI: worse in post-monsoon winter, better in monsoon)
    base_aqi = 140.0 + 80.0 * np.cos(2 * np.pi * (doy - 330) / 365.25)
    base_aqi = np.where(monsoon_mask, base_aqi - 70.0, base_aqi)
    air_quality_index = np.round(np.clip(base_aqi + rng.normal(0, 20.0, n_days), 30.0, 480.0), 1)

    # Water accumulation index (derived from rolling rainfall)
    water_accum = np.zeros(n_days)
    curr = 1.0
    for i in range(n_days):
        curr = curr * 0.88 + rainfall_mm[i] * 0.08
        water_accum[i] = curr
    water_accumulation_index = np.round(np.clip(water_accum, 0.1, 8.5), 3)

    surface_water_index = np.round(np.clip(water_accumulation_index * 0.35 + rng.normal(0, 0.05, n_days), 0.05, 1.0), 3)
    pollution_index = np.round(np.clip(air_quality_index * 0.65 + rng.normal(0, 5.0, n_days), 20.0, 350.0), 2)

    df_env = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "region": region,
        "temperature_c": temperature_c,
        "rainfall_mm": rainfall_mm,
        "humidity_percent": humidity_percent,
        "water_accumulation_index": water_accumulation_index,
        "air_quality_index": air_quality_index,
        "wind_speed_kmh": wind_speed_kmh,
        "solar_radiation_kwh_m2": solar_radiation,
        "surface_water_index": surface_water_index,
        "pollution_index": pollution_index
    })
    return df_env


def generate_disease_series(
    dates: pd.DatetimeIndex,
    region: str,
    disease: str,
    population: int,
    env_df: pd.DataFrame,
    rng: np.random.Generator
) -> pd.DataFrame:
    """Generate epidemiologically sound daily time series for a given disease and region."""
    n_days = len(dates)
    doy = dates.dayofyear.values

    # Scaling factor based on population (anchor: 2.36M for Bhopal)
    pop_scale = population / 2368145.0

    # 1. Base incidence curve & seasonal dynamics
    if disease == "Dengue":
        # Monsoon peak: August - November (DOY 210 - 320)
        seasonal_factor = np.exp(-((doy - 275) ** 2) / (2 * (38 ** 2))) * 4.5 + 0.15
        base_cases = 8.0 * pop_scale * seasonal_factor
        # Environmental modulation: rainfall and humidity lag effect
        water_accum = env_df["water_accumulation_index"].values
        base_cases *= (1.0 + 0.4 * np.roll(water_accum, 14))
        cfr_mean, hosp_mean, icu_mean = 0.004, 0.18, 0.08
    elif disease == "Malaria":
        # Moderate year-round with monsoon peak
        seasonal_factor = np.exp(-((doy - 240) ** 2) / (2 * (50 ** 2))) * 2.8 + 0.3
        base_cases = 10.0 * pop_scale * seasonal_factor
        cfr_mean, hosp_mean, icu_mean = 0.003, 0.14, 0.05
    elif disease == "COVID-19":
        # Winter waves and occasional mutation waves
        seasonal_factor = np.exp(-((doy - 15) ** 2) / (2 * (40 ** 2))) * 2.5 + 0.4
        base_cases = 18.0 * pop_scale * seasonal_factor
        cfr_mean, hosp_mean, icu_mean = 0.012, 0.22, 0.15
    elif disease == "Influenza":
        # Strict winter peak: Dec - Feb (DOY 335 - 365, 1 - 60)
        winter_dist = np.minimum(np.abs(doy - 15), 365 - np.abs(doy - 15))
        seasonal_factor = np.exp(-(winter_dist ** 2) / (2 * (35 ** 2))) * 4.0 + 0.2
        base_cases = 14.0 * pop_scale * seasonal_factor
        cfr_mean, hosp_mean, icu_mean = 0.002, 0.10, 0.04
    else:  # Typhoid
        # Post-rain water contamination
        seasonal_factor = np.exp(-((doy - 250) ** 2) / (2 * (45 ** 2))) * 2.2 + 0.35
        base_cases = 12.0 * pop_scale * seasonal_factor
        cfr_mean, hosp_mean, icu_mean = 0.005, 0.15, 0.06

    # 2. Add controlled outbreak surge events (designed to test early detection algorithms)
    surge_multiplier = np.ones(n_days)
    
    # Controlled outbreak surges:
    # Year 2023: September Dengue outbreak in Bhopal & Indore (DOY 245-275)
    # Year 2024: January Influenza spike (DOY 380-405)
    # Year 2024: October Dengue/Typhoid surge (DOY 640-670)
    # Year 2025: May-June COVID spike (DOY 860-890)
    for i, dt in enumerate(dates):
        # 2023 Monsoon vector surge
        if dt.year == 2023 and 245 <= dt.dayofyear <= 275 and disease in ["Dengue", "Malaria"]:
            surge_multiplier[i] = 2.8 + 0.8 * np.sin((dt.dayofyear - 245) / 30 * np.pi)
        # 2024 Winter respiratory surge
        elif dt.year == 2024 and 10 <= dt.dayofyear <= 45 and disease in ["Influenza", "COVID-19"]:
            surge_multiplier[i] = 3.2 + 1.1 * np.sin((dt.dayofyear - 10) / 35 * np.pi)
        # 2024 Late monsoon surge for Gwalior & Jabalpur
        elif dt.year == 2024 and 255 <= dt.dayofyear <= 285 and disease == "Dengue" and region in ["Gwalior", "Jabalpur"]:
            surge_multiplier[i] = 3.5 + 1.2 * np.sin((dt.dayofyear - 255) / 30 * np.pi)
        # 2025 Controlled multi-district surge for Dengue
        elif dt.year == 2025 and 260 <= dt.dayofyear <= 290 and disease == "Dengue":
            surge_multiplier[i] = 2.6 + 0.9 * np.sin((dt.dayofyear - 260) / 30 * np.pi)

    # Daily stochastic realization
    lambda_daily = np.maximum(base_cases * surge_multiplier, 0.5)
    new_cases = rng.poisson(lambda_daily)

    # Active cases (rolling accumulation over 14 days) and recoveries
    active_cases = np.zeros(n_days, dtype=int)
    recovered = np.zeros(n_days, dtype=int)
    deaths = np.zeros(n_days, dtype=int)
    hospitalizations = np.zeros(n_days, dtype=int)
    icu_admissions = np.zeros(n_days, dtype=int)
    tests = np.zeros(n_days, dtype=int)
    tpr = np.zeros(n_days, dtype=float)

    curr_active = int(new_cases[0] * 3)
    for i in range(n_days):
        c = new_cases[i]
        # Deaths
        d = rng.binomial(c, np.clip(cfr_mean * rng.uniform(0.8, 1.2), 0.0, 0.2))
        deaths[i] = d

        # Hospitalizations
        h = rng.binomial(c, np.clip(hosp_mean * rng.uniform(0.85, 1.15), 0.0, 0.8))
        hospitalizations[i] = h

        # ICU admissions
        icu = rng.binomial(h, np.clip(icu_mean * rng.uniform(0.8, 1.2), 0.0, 0.5)) if h > 0 else 0
        icu_admissions[i] = icu

        # Daily recoveries (approx 1/12th of active cases resolved per day)
        rec = int(np.clip(rng.binomial(max(curr_active, 1), 0.08), 0, curr_active))
        recovered[i] = rec

        # Active case update
        curr_active = max(0, curr_active + c - rec - d)
        active_cases[i] = curr_active

        # Testing counts (positive tests <= total tests)
        t_base = int(max(c * rng.uniform(12, 28) + rng.normal(300, 50), c + 50))
        tests[i] = t_base
        tpr[i] = round(float(c) / float(t_base) * 100.0, 2)

    df_epi = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "region": region,
        "disease": disease,
        "new_cases": new_cases,
        "active_cases": active_cases,
        "recovered": recovered,
        "deaths": deaths,
        "hospitalizations": hospitalizations,
        "icu_admissions": icu_admissions,
        "tests": tests,
        "test_positivity_rate_percent": tpr,
        "data_status": "SYNTHETIC_DEMO"
    })
    return df_epi


def generate_healthcare_monthly(
    start_date: datetime,
    end_date: datetime,
    region: str,
    population: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """Generate monthly healthcare capacity and readiness indicators."""
    months = pd.date_range(start_date, end_date, freq="MS")
    n_months = len(months)
    pop_100k = population / 100000.0

    # Benchmark: ~250 beds per 100k, ~18 ICU beds per 100k, ~110 doctors per 100k
    base_beds = int(pop_100k * rng.uniform(240, 280))
    base_icu = int(base_beds * rng.uniform(0.06, 0.08))
    base_doctors = int(pop_100k * rng.uniform(105, 130))
    base_hospitals = int(pop_100k * rng.uniform(3.5, 4.5))

    records = []
    for m in months:
        # Slight gradual capacity growth over time
        growth_factor = 1.0 + (m.year - start_date.year) * 0.03
        h_count = int(base_hospitals * growth_factor)
        beds = int(base_beds * growth_factor + rng.integers(-20, 20))
        icu = int(base_icu * growth_factor + rng.integers(-3, 5))
        docs = int(base_doctors * growth_factor + rng.integers(-15, 25))
        test_cap = int(pop_100k * rng.uniform(220, 320))
        avail_o2 = round(float(np.clip(rng.normal(82.0, 4.5), 60.0, 99.0)), 1)
        med_avail = round(float(np.clip(rng.normal(86.0, 3.5), 70.0, 98.0)), 1)
        emerg = int(pop_100k * rng.uniform(1.2, 1.8))
        hosp_mort = round(float(np.clip(rng.normal(1.65, 0.25), 0.8, 3.5)), 2)
        ambulances = int(pop_100k * rng.uniform(2.0, 3.2))

        records.append({
            "month": m.strftime("%Y-%m-%d"),
            "region": region,
            "hospital_count": h_count,
            "hospital_beds": beds,
            "available_oxygen_percent": avail_o2,
            "icu_beds": icu,
            "doctors": docs,
            "testing_capacity_tests_per_day": test_cap,
            "medicine_availability_percent": med_avail,
            "emergency_units": emerg,
            "hospital_mortality_rate_percent": hosp_mort,
            "ambulances_available": ambulances,
            "data_status": "SYNTHETIC_DEMO"
        })

    return pd.DataFrame(records)


def run_generate_all_demo_data(
    start_date_str: str = "2023-01-01",
    end_date_str: str = "2025-12-31",
    output_dir: str = "data/demo",
    seed: int = 42
) -> Dict[str, Any]:
    """Execute complete generation of demonstration datasets."""
    rng = np.random.default_rng(seed)
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = pd.date_range(start_date, end_date, freq="D")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating demonstration data from {start_date_str} to {end_date_str} for 5 regions...")

    # 1. Demographics
    demo_df = pd.DataFrame(list(DEMOGRAPHIC_ANCHORS.values()))
    demo_csv = out_path / "01_demographics_demo.csv"
    demo_df.to_csv(demo_csv, index=False)
    print(f"  [+] Demographics saved: {demo_csv} ({len(demo_df)} regions)")

    # 2. Environmental & Epidemiological
    all_env = []
    all_epi = []
    all_hc = []

    for region, anchor in DEMOGRAPHIC_ANCHORS.items():
        pop = anchor["population_2011"]
        # Environmental
        env_df = generate_environmental_series(dates, region, rng)
        all_env.append(env_df)

        # Epidemiological for each disease
        for disease in DISEASES:
            epi_df = generate_disease_series(dates, region, disease, pop, env_df, rng)
            all_epi.append(epi_df)

        # Healthcare monthly
        hc_df = generate_healthcare_monthly(start_date, end_date, region, pop, rng)
        all_hc.append(hc_df)

    combined_env = pd.concat(all_env, ignore_index=True)
    env_csv = out_path / "02_environmental_daily_demo.csv"
    combined_env.to_csv(env_csv, index=False)
    print(f"  [+] Environmental saved: {env_csv} ({len(combined_env)} rows)")

    combined_epi = pd.concat(all_epi, ignore_index=True)
    epi_csv = out_path / "05_epidemiological_daily_demo.csv"
    combined_epi.to_csv(epi_csv, index=False)
    print(f"  [+] Epidemiological saved: {epi_csv} ({len(combined_epi)} rows)")

    combined_hc = pd.concat(all_hc, ignore_index=True)
    hc_csv = out_path / "03_healthcare_monthly_demo.csv"
    combined_hc.to_csv(hc_csv, index=False)
    print(f"  [+] Healthcare monthly saved: {hc_csv} ({len(combined_hc)} rows)")

    # Provenance metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "date_range": [start_date_str, end_date_str],
        "regions": list(DEMOGRAPHIC_ANCHORS.keys()),
        "diseases": DISEASES,
        "seed": seed,
        "provenance": {
            "demographics": "OFFICIAL_STATIC_REFERENCE (Census 2011 anchors)",
            "epidemiological": "SYNTHETIC_DEMO (Controlled biological & seasonal models)",
            "environmental": "SYNTHETIC_DEMO (Central India meteorological calibration)",
            "healthcare": "SYNTHETIC_DEMO (WHO & NITI Aayog capacity ratios)"
        },
        "disclaimer": "DEMO MODE — Synthetic epidemiological data. Results are illustrative and are not live public-health surveillance."
    }

    meta_file = out_path / "metadata_provenance.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [+] Provenance metadata saved: {meta_file}")
    return metadata


if __name__ == "__main__":
    run_generate_all_demo_data()
