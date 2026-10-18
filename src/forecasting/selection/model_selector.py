"""Model Selector (Selects best model per region + disease + horizon)."""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("model_selector")


def select_best_models(
    val_results: List[Dict[str, Any]],
    output_dir: Path,
    primary_metric: str = "MAE",
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, int], str]]:
    """Selects the best model separately for each (region, disease, horizon) combination.

    Rule:
    -----
    Primary Selection Metric: MAE (lowest error).
    Secondary: RMSE, SMAPE.

    Returns
    -------
    selected_df : pd.DataFrame summary of winning models
    winning_models_map : Dict mapping (region, disease, horizon) -> winning_model_name
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(val_results)

    if df.empty:
        logger.warning("Empty validation results list. Model selection skipped.")
        return pd.DataFrame(), {}

    # Sort by primary metric ascending
    df = df.sort_values(["region", "disease", "horizon", primary_metric, "RMSE"]).reset_index(drop=True)

    # Pick top row per (region, disease, horizon)
    winning_df = df.groupby(["region", "disease", "horizon"], as_index=False).first()

    winning_map = {}
    for _, row in winning_df.iterrows():
        key = (row["region"], row["disease"], int(row["horizon"]))
        winning_map[key] = row["model"]
        logger.info(
            f"Selected best model for [{row['region']}-{row['disease']}-{row['horizon']}d]: "
            f"{row['model']} (Val {primary_metric}={row[primary_metric]:.4f})"
        )

    out_csv = output_dir / "selected_models.csv"
    winning_df.to_csv(out_csv, index=False)
    logger.info(f"Selected models saved to {out_csv}")

    return winning_df, winning_map
