"""EpiWatch Phase 3 — Severity Assigner & Contributing Signals Generator.

Evaluates disease outbreak severity independently from anomaly detection:
  - Case volume and growth (new_cases, growth_rate_7d)
  - Population impact (incidence_per_100k)
  - Healthcare burden (hospitalizations, icu_admissions, hospitalization_rate)
  - Mortality burden (deaths, case_fatality_rate)

Outputs:
--------
- severity: "LOW", "MODERATE", "HIGH", "CRITICAL"
- contributing_signals: JSON-formatted list of deterministic human-readable explanations

Design Principle:
-----------------
Severity is kept strictly distinct from anomaly detection. A rare anomaly with 5 cases
may be an anomaly (high anomaly score) but LOW severity. Conversely, a seasonal peak
with 1,000 cases and high hospitalisations may have HIGH severity.
"""

import json
from typing import List
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("severity")


def assign_severity_and_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Compute outbreak severity tiers and generate human-readable contributing signals.

    Parameters
    ----------
    df : DataFrame with epidemiological, healthcare, and baseline columns

    Returns
    -------
    out_df : DataFrame with severity and contributing_signals columns
    """
    logger.info("Computing Outbreak Severity and generating deterministic contributing signals...")
    out_df = df.copy()

    severities: List[str] = []
    signals_list: List[str] = []

    for idx, row in out_df.iterrows():
        reasons = []

        cases = float(row.get("new_cases", 0))
        growth = float(row.get("growth_rate_7d", 0))
        inc = float(row.get("incidence_per_100k", 0))
        hosp = float(row.get("hospitalizations", 0))
        icu = float(row.get("icu_admissions", 0))
        deaths = float(row.get("deaths", 0))
        dev = float(row.get("baseline_deviation", 0))
        status = str(row.get("outbreak_status", "NORMAL"))

        # Severity Score Points (0–100 scale)
        points = 0.0

        if cases > 100:
            points += 15
        elif cases > 50:
            points += 10

        if growth > 30.0:
            points += 20
            reasons.append(f"Rapid 7-day case growth ({growth:.1f}%)")
        elif growth > 10.0:
            points += 10
            reasons.append(f"Elevated 7-day case growth ({growth:.1f}%)")

        if inc > 5.0:
            points += 20
            reasons.append(f"High population incidence rate ({inc:.2f} per 100k)")
        elif inc > 2.0:
            points += 10
            reasons.append(f"Moderate population incidence rate ({inc:.2f} per 100k)")

        if hosp > 20:
            points += 20
            reasons.append(f"High daily hospital admissions ({int(hosp)} patients)")
        elif hosp > 5:
            points += 10

        if icu > 5:
            points += 15
            reasons.append(f"ICU capacity stress ({int(icu)} admissions)")

        if deaths >= 3:
            points += 15
            reasons.append(f"Elevated mortality recorded ({int(deaths)} deaths)")

        if dev >= 2.0:
            reasons.append(f"Cases are significantly above historical baseline (+{dev:.1f}σ)")

        # Map points to severity
        if points >= 60:
            sev = "CRITICAL"
        elif points >= 40:
            sev = "HIGH"
        elif points >= 20:
            sev = "MODERATE"
        else:
            sev = "LOW"

        if not reasons:
            reasons.append("Disease activity within expected baseline parameters")

        severities.append(sev)
        signals_list.append(json.dumps(reasons))

    out_df["severity"] = severities
    out_df["contributing_signals"] = signals_list

    counts = out_df["severity"].value_counts()
    logger.info(
        f"Severity Assignment Complete — LOW: {counts.get('LOW', 0)} | "
        f"MODERATE: {counts.get('MODERATE', 0)} | HIGH: {counts.get('HIGH', 0)} | "
        f"CRITICAL: {counts.get('CRITICAL', 0)}"
    )

    return out_df
