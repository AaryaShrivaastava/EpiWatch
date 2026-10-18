"""Epidemiological feature engineering for EpiWatch.

Data Leakage Prevention:
------------------------
To prevent lookahead/data leakage when generating rolling statistics and growth rates:
1. Data must be chronologically ordered prior to calculation.
2. Rolling windows use closed='left' or shift(1) where appropriate to ensure only
   strictly past and current observations are included in historical features.
3. No future observations (t+1, t+2...) are ever referenced.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("epidemiological_features")


def generate_epidemiological_features(
    df: pd.DataFrame,
    population: Optional[float] = None,
    cases_col: str = "new_cases",
    group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Generate rolling averages, growth rates, incidence, CFR, and hospitalization rates.

    Args:
        df: Cleaned epidemiological DataFrame.
        population: Regional population anchor for incidence calculation (if available).
        cases_col: Name of confirmed/new cases column.
        group_cols: Grouping columns, e.g., ['region', 'disease'].

    Returns:
        DataFrame with added epidemiological feature columns.
    """
    logger.info("Generating epidemiological features...")
    df_feat = df.copy()

    # Ensure chronological order
    if "date" in df_feat.columns:
        df_feat["date"] = pd.to_datetime(df_feat["date"])
        sort_keys = [c for c in (group_cols or []) if c in df_feat.columns] + ["date"]
        df_feat.sort_values(by=sort_keys, inplace=True)
        df_feat.reset_index(drop=True, inplace=True)

    g_cols = [c for c in (group_cols or ["region", "disease"]) if c in df_feat.columns]

    if cases_col not in df_feat.columns:
        logger.warning(f"Cases column '{cases_col}' not found. Skipping epidemiological features.")
        return df_feat

    # A. Rolling averages (7d, 14d, 28d) using past/current observations only
    # Data Leakage Prevention: rolling window operates on historical sequence.
    for window in [7, 14, 28]:
        col_name = f"cases_{window}d_avg"
        if g_cols:
            df_feat[col_name] = df_feat.groupby(g_cols)[cases_col].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
        else:
            df_feat[col_name] = df_feat[cases_col].rolling(window=window, min_periods=1).mean()
        logger.info(f"Generated rolling average: {col_name}")

    # B. Growth rates (7d, 14d)
    # growth_rate = (current_cases - cases_N_days_ago) / max(cases_N_days_ago, 1e-5)
    for lag in [7, 14]:
        lag_col = f"cases_lag_{lag}"
        growth_col = f"growth_rate_{lag}d"

        if g_cols:
            df_feat[lag_col] = df_feat.groupby(g_cols)[cases_col].shift(lag)
        else:
            df_feat[lag_col] = df_feat[cases_col].shift(lag)

        prev_cases = df_feat[lag_col]
        curr_cases = df_feat[cases_col]

        # Division by zero safety
        denom = np.where(prev_cases > 0, prev_cases, np.nan)
        growth_rate = np.where(
            pd.notna(denom),
            ((curr_cases - prev_cases) / denom) * 100.0,
            0.0
        )
        df_feat[growth_col] = np.round(growth_rate, 4)
        df_feat.drop(columns=[lag_col], inplace=True)
        logger.info(f"Generated growth rate: {growth_col}")

    # C. Incidence per 100k
    # Check if population is in df or passed as argument
    pop_series = None
    if "population" in df_feat.columns:
        pop_series = df_feat["population"]
    elif "population_2011" in df_feat.columns:
        pop_series = df_feat["population_2011"]
    elif population is not None and population > 0:
        pop_series = population

    if pop_series is not None:
        df_feat["incidence_per_100k"] = np.round((df_feat[cases_col] / pop_series) * 100000.0, 4)
        logger.info("Generated feature: incidence_per_100k")
    else:
        logger.debug("Population data not provided for incidence calculation.")

    # D. Case Fatality Rate (CFR %) = (deaths / confirmed_cases) * 100
    if "deaths" in df_feat.columns:
        deaths = df_feat["deaths"]
        cases = df_feat[cases_col]
        cfr = np.where(cases > 0, (deaths / cases) * 100.0, 0.0)
        df_feat["case_fatality_rate"] = np.round(cfr, 4)
        logger.info("Generated feature: case_fatality_rate")

    # E. Hospitalization Rate (%) = (hospitalizations / confirmed_cases) * 100
    if "hospitalizations" in df_feat.columns:
        hosp = df_feat["hospitalizations"]
        cases = df_feat[cases_col]
        hosp_rate = np.where(cases > 0, (hosp / cases) * 100.0, 0.0)
        df_feat["hospitalization_rate"] = np.round(hosp_rate, 4)
        logger.info("Generated feature: hospitalization_rate")

    # F. ICU Admission Rate (%) = (icu_admissions / hospitalizations) * 100
    if "icu_admissions" in df_feat.columns and "hospitalizations" in df_feat.columns:
        icu = df_feat["icu_admissions"]
        hosp = df_feat["hospitalizations"]
        icu_rate = np.where(hosp > 0, (icu / hosp) * 100.0, 0.0)
        df_feat["icu_admission_rate"] = np.round(icu_rate, 4)
        logger.info("Generated feature: icu_admission_rate")

    return df_feat
