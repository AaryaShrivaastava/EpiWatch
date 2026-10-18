"""Healthcare feature engineering for EpiWatch."""

from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("healthcare_features")


def generate_healthcare_features(
    df: pd.DataFrame,
    population: Optional[float] = None
) -> pd.DataFrame:
    """Calculate normalized healthcare metrics per 100k population and occupancy/utilization ratios.

    Args:
        df: Cleaned healthcare DataFrame (or combined dataframe).
        population: Regional population anchor.

    Returns:
        DataFrame with added healthcare normalized features.
    """
    logger.info("Generating healthcare normalized features...")
    df_feat = df.copy()

    pop_val = None
    if "population" in df_feat.columns:
        pop_val = df_feat["population"]
    elif "population_2011" in df_feat.columns:
        pop_val = df_feat["population_2011"]
    elif population is not None and population > 0:
        pop_val = population

    # 1. Beds, ICU beds, Doctors per 100k
    if pop_val is not None:
        if "hospital_beds" in df_feat.columns:
            df_feat["beds_per_100k"] = np.round((df_feat["hospital_beds"] / pop_val) * 100000.0, 4)
            logger.info("Generated feature: beds_per_100k")

        if "icu_beds" in df_feat.columns:
            df_feat["icu_beds_per_100k"] = np.round((df_feat["icu_beds"] / pop_val) * 100000.0, 4)
            logger.info("Generated feature: icu_beds_per_100k")

        if "doctors" in df_feat.columns:
            df_feat["doctors_per_100k"] = np.round((df_feat["doctors"] / pop_val) * 100000.0, 4)
            logger.info("Generated feature: doctors_per_100k")
    else:
        logger.debug("Population not available for per-100k healthcare metrics.")

    # 2. Bed Occupancy Ratio
    if "hospitalizations" in df_feat.columns and "hospital_beds" in df_feat.columns:
        beds = df_feat["hospital_beds"]
        hosp = df_feat["hospitalizations"]
        ratio = np.where(beds > 0, np.minimum(hosp / beds, 1.0), np.nan)
        df_feat["bed_occupancy_ratio"] = np.round(ratio, 4)
        logger.info("Generated feature: bed_occupancy_ratio")
    elif "bed_occupancy" in df_feat.columns:
        df_feat["bed_occupancy_ratio"] = df_feat["bed_occupancy"]

    # 3. ICU Occupancy Ratio
    if "icu_admissions" in df_feat.columns and "icu_beds" in df_feat.columns:
        icu_b = df_feat["icu_beds"]
        icu_a = df_feat["icu_admissions"]
        ratio = np.where(icu_b > 0, np.minimum(icu_a / icu_b, 1.0), np.nan)
        df_feat["icu_occupancy_ratio"] = np.round(ratio, 4)
        logger.info("Generated feature: icu_occupancy_ratio")

    # 4. Testing Utilization Ratio = tests / testing_capacity
    test_cap_col = (
        "testing_capacity_tests_per_day"
        if "testing_capacity_tests_per_day" in df_feat.columns
        else ("testing_capacity" if "testing_capacity" in df_feat.columns else None)
    )
    if "tests" in df_feat.columns and test_cap_col:
        cap = df_feat[test_cap_col]
        tests = df_feat["tests"]
        utilization = np.where(cap > 0, tests / cap, np.nan)
        df_feat["testing_utilization"] = np.round(utilization, 4)
        logger.info("Generated feature: testing_utilization")

    return df_feat
