"""Epidemiological metrics module for EpiWatch Phase 2.

Handles Disease Summary, Incidence Analysis, Case Fatality Rate (CFR),
Hospitalization Analysis, and Test Positivity Analysis.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("epidemiological_metrics")


def calculate_disease_summary(
    df_epi: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Calculate overall disease summary statistics per disease.

    Metrics:
    - total_cases
    - avg_weekly_cases
    - max_weekly_cases
    - min_weekly_cases
    - total_deaths
    - total_hospitalizations
    - hospitalization_rate_percent
    - case_fatality_rate_percent
    - avg_positivity_percent
    - peak_cases
    - peak_date

    Saves: data/analytics/disease_summary.csv
    """
    logger.info("Calculating disease overview summary metrics...")
    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])

    summary_list = []
    diseases = df["disease"].unique()

    for d in diseases:
        df_d = df[df["disease"] == d].sort_values("date")

        total_cases = int(df_d["new_cases"].sum())
        total_deaths = int(df_d["deaths"].sum()) if "deaths" in df_d.columns else 0
        total_hosp = int(df_d["hospitalizations"].sum()) if "hospitalizations" in df_d.columns else 0

        # Hospitalization rate % = total_hosp / total_cases * 100
        hosp_rate = round(float((total_hosp / total_cases * 100.0) if total_cases > 0 else np.nan), 2)

        # CFR % = total_deaths / total_cases * 100
        cfr = round(float((total_deaths / total_cases * 100.0) if total_cases > 0 else np.nan), 2)

        # Test positivity average
        pos_series = df_d["test_positivity_rate_percent"] if "test_positivity_rate_percent" in df_d.columns else None
        avg_pos = round(float(pos_series.mean()), 2) if pos_series is not None and not pos_series.dropna().empty else np.nan

        # Peak date and count
        peak_row = df_d.loc[df_d["new_cases"].idxmax()]
        peak_cases = int(peak_row["new_cases"])
        peak_date = str(pd.to_datetime(peak_row["date"]).strftime("%Y-%m-%d"))

        # Weekly case aggregation
        df_d_weekly = df_d.set_index("date").resample("W-MON")["new_cases"].sum()
        avg_weekly = round(float(df_d_weekly.mean()), 2)
        max_weekly = int(df_d_weekly.max())
        min_weekly = int(df_d_weekly.min())
        peak_week = str(df_d_weekly.idxmax().strftime("%Y-W%U")) if not df_d_weekly.empty else None

        summary_list.append({
            "disease": d,
            "total_cases": total_cases,
            "avg_weekly_cases": avg_weekly,
            "max_weekly_cases": max_weekly,
            "min_weekly_cases": min_weekly,
            "total_deaths": total_deaths,
            "total_hospitalizations": total_hosp,
            "hospitalization_rate_percent": hosp_rate,
            "case_fatality_rate_percent": cfr,
            "avg_positivity_percent": avg_pos,
            "peak_cases": peak_cases,
            "peak_date": peak_date,
            "peak_week": peak_week
        })

    summary_df = pd.DataFrame(summary_list)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "disease_summary.csv"
    summary_df.to_csv(csv_file, index=False)
    logger.info(f"Saved disease summary CSV: {csv_file}")
    return summary_df


def calculate_incidence(
    df_epi: pd.DataFrame,
    population: float,
    output_dir: Path
) -> pd.DataFrame:
    """Calculate weekly, monthly, and yearly incidence per 100,000 population.

    Formula: incidence_per_100k = (new_cases / population) * 100,000

    Saves: data/analytics/incidence_analysis.csv
    """
    logger.info("Calculating incidence analysis metrics...")
    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])

    incidence_rows = []
    for d, df_d in df.groupby("disease"):
        df_d = df_d.set_index("date").sort_index()

        # Weekly incidence
        w_cases = df_d["new_cases"].resample("W-MON").sum()
        # Monthly incidence
        m_cases = df_d["new_cases"].resample("ME").sum()
        # Yearly incidence
        y_cases = df_d["new_cases"].resample("YE").sum()

        for date_val, cases in w_cases.items():
            inc = round(float((cases / population) * 100000.0), 4)
            incidence_rows.append({
                "disease": d,
                "period": "weekly",
                "date": date_val.strftime("%Y-%m-%d"),
                "raw_cases": int(cases),
                "incidence_per_100k": inc
            })

        for date_val, cases in m_cases.items():
            inc = round(float((cases / population) * 100000.0), 4)
            incidence_rows.append({
                "disease": d,
                "period": "monthly",
                "date": date_val.strftime("%Y-%m"),
                "raw_cases": int(cases),
                "incidence_per_100k": inc
            })

        for date_val, cases in y_cases.items():
            inc = round(float((cases / population) * 100000.0), 4)
            incidence_rows.append({
                "disease": d,
                "period": "yearly",
                "date": str(date_val.year),
                "raw_cases": int(cases),
                "incidence_per_100k": inc
            })

    inc_df = pd.DataFrame(incidence_rows)
    csv_file = output_dir / "incidence_analysis.csv"
    inc_df.to_csv(csv_file, index=False)
    logger.info(f"Saved incidence analysis CSV: {csv_file}")
    return inc_df


def calculate_cfr(
    df_epi: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Calculate disease-level, yearly, and monthly Case Fatality Rate (CFR %).

    Formula: CFR = (deaths / confirmed_cases) * 100
    Returns NaN when denominator (cases) is 0 to avoid misleading 0.0 values.

    Saves: data/analytics/cfr_analysis.csv
    """
    logger.info("Calculating Case Fatality Rate (CFR) metrics...")
    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])

    cfr_rows = []
    for d, df_d in df.groupby("disease"):
        df_d = df_d.set_index("date").sort_index()

        # Overall
        tot_c = df_d["new_cases"].sum()
        tot_d = df_d["deaths"].sum() if "deaths" in df_d.columns else 0
        overall_cfr = (tot_d / tot_c * 100.0) if tot_c > 0 else np.nan
        cfr_rows.append({
            "disease": d,
            "period": "overall",
            "date": "all",
            "cases": int(tot_c),
            "deaths": int(tot_d),
            "cfr_percent": round(float(overall_cfr), 4) if pd.notna(overall_cfr) else np.nan
        })

        # Yearly
        y_df = df_d.resample("YE").agg({"new_cases": "sum", "deaths": "sum"})
        for date_val, row in y_df.iterrows():
            c, d_cnt = row["new_cases"], row["deaths"]
            y_cfr = (d_cnt / c * 100.0) if c > 0 else np.nan
            cfr_rows.append({
                "disease": d,
                "period": "yearly",
                "date": str(date_val.year),
                "cases": int(c),
                "deaths": int(d_cnt),
                "cfr_percent": round(float(y_cfr), 4) if pd.notna(y_cfr) else np.nan
            })

    cfr_df = pd.DataFrame(cfr_rows)
    csv_file = output_dir / "cfr_analysis.csv"
    cfr_df.to_csv(csv_file, index=False)
    logger.info(f"Saved CFR analysis CSV: {csv_file}")
    return cfr_df


def calculate_hospitalizations(
    df_epi: pd.DataFrame,
    population: float,
    output_dir: Path
) -> pd.DataFrame:
    """Calculate hospitalization rates and hospitalizations per 100k per disease.

    Saves: data/analytics/hospitalization_analysis.csv
    """
    logger.info("Calculating hospitalization analysis metrics...")
    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])

    hosp_rows = []
    for d, df_d in df.groupby("disease"):
        df_d = df_d.set_index("date").sort_index()
        m_df = df_d.resample("ME").agg({
            "new_cases": "sum",
            "hospitalizations": "sum",
            "deaths": "sum"
        })

        for date_val, row in m_df.iterrows():
            c, h, d_cnt = row["new_cases"], row["hospitalizations"], row["deaths"]
            h_rate = (h / c * 100.0) if c > 0 else np.nan
            h_100k = (h / population) * 100000.0

            hosp_rows.append({
                "disease": d,
                "month": date_val.strftime("%Y-%m"),
                "cases": int(c),
                "hospitalizations": int(h),
                "deaths": int(d_cnt),
                "hospitalization_rate_percent": round(float(h_rate), 4) if pd.notna(h_rate) else np.nan,
                "hospitalizations_per_100k": round(float(h_100k), 4)
            })

    hosp_df = pd.DataFrame(hosp_rows)
    csv_file = output_dir / "hospitalization_analysis.csv"
    hosp_df.to_csv(csv_file, index=False)
    logger.info(f"Saved hospitalization analysis CSV: {csv_file}")
    return hosp_df


def calculate_positivity(
    df_epi: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Calculate test positivity rate metrics per disease and month.

    Formula: positivity_rate = (confirmed_cases / tested_cases) * 100

    Saves: data/analytics/positivity_analysis.csv
    """
    logger.info("Calculating test positivity analysis metrics...")
    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])

    pos_rows = []
    if "tests" in df.columns:
        for d, df_d in df.groupby("disease"):
            df_d = df_d.set_index("date").sort_index()
            m_df = df_d.resample("ME").agg({"new_cases": "sum", "tests": "sum"})

            for date_val, row in m_df.iterrows():
                c, t = row["new_cases"], row["tests"]
                pos = (c / t * 100.0) if t > 0 else np.nan
                pos_rows.append({
                    "disease": d,
                    "month": date_val.strftime("%Y-%m"),
                    "cases": int(c),
                    "tests": int(t),
                    "positivity_rate_percent": round(float(pos), 4) if pd.notna(pos) else np.nan
                })

    pos_df = pd.DataFrame(pos_rows)
    csv_file = output_dir / "positivity_analysis.csv"
    pos_df.to_csv(csv_file, index=False)
    logger.info(f"Saved positivity analysis CSV: {csv_file}")
    return pos_df
