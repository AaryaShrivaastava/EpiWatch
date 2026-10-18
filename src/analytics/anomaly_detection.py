"""Statistical anomaly detection and candidate outbreak episode grouping.

Important Terminology:
----------------------
The anomalies identified in this module are POTENTIAL STATISTICAL ANOMALIES computed using
rolling statistical thresholds (rolling mean + 2 * rolling std). They represent statistical candidate
episodes for analytical evaluation and baseline benchmarking, NOT medically or officially confirmed outbreaks.
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("anomaly_detection")


def detect_potential_anomalies(
    df_epi: pd.DataFrame,
    output_dir: Path,
    config_path: str = "config.yaml"
) -> pd.DataFrame:
    """Detect statistical anomalies using transparent rolling thresholds.

    Formula:
    baseline = rolling 28-day mean
    upper_threshold = baseline + (2.0 * rolling 28-day std)
    potential_anomaly = 1 if new_cases > upper_threshold else 0

    Saves: data/analytics/potential_anomalies.csv
    """
    logger.info("Detecting potential statistical anomalies...")
    config = get_config(config_path)
    window = config.get("anomaly_detection.rolling_window_days", 28)
    multiplier = config.get("anomaly_detection.std_multiplier", 2.0)

    df = df_epi.copy()
    df["date"] = pd.to_datetime(df["date"])

    anomaly_rows = []
    for d, df_d in df.groupby("disease"):
        df_d = df_d.sort_values("date").copy()

        # Rolling baseline and upper threshold (past window only)
        rolling_mean = df_d["new_cases"].rolling(window=window, min_periods=7).mean()
        rolling_std = df_d["new_cases"].rolling(window=window, min_periods=7).std().fillna(0.0)

        df_d["baseline_mean"] = np.round(rolling_mean, 2)
        df_d["upper_threshold"] = np.round(rolling_mean + (multiplier * rolling_std), 2)

        # Flag potential anomaly
        df_d["potential_anomaly"] = np.where(
            (df_d["new_cases"] > df_d["upper_threshold"]) & (df_d["upper_threshold"] > 0),
            1,
            0
        )

        anomaly_rows.append(df_d)

    anom_df = pd.concat(anomaly_rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "potential_anomalies.csv"
    anom_df.to_csv(csv_file, index=False)
    logger.info(f"Saved potential statistical anomalies CSV: {csv_file}")
    return anom_df


def group_outbreak_episodes(
    anom_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Group consecutive potential statistical anomalies into candidate outbreak episodes.

    Calculates:
    - episode_id
    - disease
    - start_date
    - end_date
    - duration_days
    - peak_cases
    - total_cases
    - peak_growth_rate

    Saves: data/analytics/potential_outbreak_episodes.csv
    """
    logger.info("Grouping potential statistical anomalies into candidate outbreak episodes...")
    df = anom_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    episodes = []
    episode_counter = 1

    for d, df_d in df.groupby("disease"):
        df_d = df_d.sort_values("date").reset_index(drop=True)

        # Identify contiguous blocks of potential_anomaly == 1
        df_d["block"] = (df_d["potential_anomaly"] != df_d["potential_anomaly"].shift(1)).cumsum()
        anomaly_blocks = df_d[df_d["potential_anomaly"] == 1]

        for block_id, group in anomaly_blocks.groupby("block"):
            start_d = group["date"].min()
            end_d = group["date"].max()
            duration = (end_d - start_d).days + 1
            peak_c = int(group["new_cases"].max())
            total_c = int(group["new_cases"].sum())

            peak_growth = round(float(group["growth_rate_7d"].max()), 2) if "growth_rate_7d" in group.columns else np.nan

            episodes.append({
                "episode_id": f"EP_{episode_counter:04d}",
                "disease": d,
                "start_date": start_d.strftime("%Y-%m-%d"),
                "end_date": end_d.strftime("%Y-%m-%d"),
                "duration_days": duration,
                "peak_cases": peak_c,
                "total_cases": total_c,
                "peak_growth_rate_percent": peak_growth
            })
            episode_counter += 1

    episodes_df = pd.DataFrame(episodes)
    csv_file = output_dir / "potential_outbreak_episodes.csv"
    episodes_df.to_csv(csv_file, index=False)
    logger.info(f"Saved candidate outbreak episodes CSV: {csv_file}")
    return episodes_df
