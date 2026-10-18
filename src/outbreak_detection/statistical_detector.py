"""EpiWatch Phase 3 — Level 2: Statistical Signal Combiner.

Extracts and normalises 5 multi-dimensional epidemiological signals:
  A. Case Spike Ratio       : current_cases / max(1.0, baseline_mean)
  B. Growth Rate Signal     : growth_rate_7d (% change over 7 days)
  C. Incidence Rate Signal  : incidence_per_100k
  D. Positivity Rate Signal : test_positivity_rate_percent
  E. Hospitalisation Signal : hospitalization_rate (% of cases hospitalised)

Each signal is min-max scaled or clipped to a [0, 1] range, then combined into a
composite `statistical_signal_score` (0–100).
"""

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("statistical_detector")


def _min_max_norm(series: pd.Series, max_clip: float) -> pd.Series:
    """Helper to clip and scale a numerical series into [0, 1]."""
    clipped = np.clip(series.fillna(0.0), 0.0, max_clip)
    return clipped / max_clip if max_clip > 0 else pd.Series(0.0, index=series.index)


def compute_statistical_signals(
    df: pd.DataFrame,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    """Compute and combine Level 2 statistical epidemiological signals.

    Parameters
    ----------
    df          : DataFrame with baseline_mean, growth_rate_7d, incidence_per_100k, etc.
    config_path : Path to config.yaml

    Returns
    -------
    out_df : DataFrame with individual normalized signals and composite statistical_signal_score
    """
    logger.info("Computing Level 2 Multi-Dimensional Statistical Signals...")
    out_df = df.copy()

    # 1. Case Spike Ratio (current_cases / baseline_mean)
    b_mean = out_df.get("baseline_mean", out_df["new_cases"].expanding().mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        spike_ratio = np.where(b_mean > 0.5, out_df["new_cases"] / b_mean, 1.0)
    out_df["case_spike_ratio"] = np.round(spike_ratio, 2)
    # Spike ratio of 3.0+ = 1.0 signal score
    out_df["case_spike_signal"] = _min_max_norm(out_df["case_spike_ratio"] - 1.0, max_clip=2.0)

    # 2. Growth Signal (growth_rate_7d)
    growth_7d = out_df.get("growth_rate_7d", pd.Series(0.0, index=out_df.index))
    # Growth of 50%+ = 1.0 signal score
    out_df["growth_signal"] = _min_max_norm(growth_7d, max_clip=50.0)

    # 3. Incidence Signal (incidence_per_100k)
    inc = out_df.get("incidence_per_100k", pd.Series(0.0, index=out_df.index))
    # Incidence of 10 per 100k = 1.0 signal score
    out_df["incidence_signal"] = _min_max_norm(inc, max_clip=10.0)

    # 4. Positivity Rate Signal (test_positivity_rate_percent)
    pos = out_df.get("test_positivity_rate_percent", pd.Series(0.0, index=out_df.index))
    # Positivity of 20%+ = 1.0 signal score
    out_df["positivity_signal"] = _min_max_norm(pos, max_clip=20.0)

    # 5. Hospitalisation Signal (hospitalization_rate)
    hosp = out_df.get("hospitalization_rate", pd.Series(0.0, index=out_df.index))
    # Hospitalisation rate of 25%+ = 1.0 signal score
    out_df["hospitalization_signal"] = _min_max_norm(hosp, max_clip=25.0)

    # Composite weighted score (0–100)
    comp_score = (
        0.30 * out_df["case_spike_signal"] +
        0.25 * out_df["growth_signal"] +
        0.20 * out_df["incidence_signal"] +
        0.15 * out_df["positivity_signal"] +
        0.10 * out_df["hospitalization_signal"]
    ) * 100.0

    out_df["statistical_signal_score"] = np.round(comp_score, 2)
    logger.info(f"Level 2 Statistical Signals computed — Mean composite score: {out_df['statistical_signal_score'].mean():.2f}")

    return out_df
