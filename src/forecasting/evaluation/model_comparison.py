"""Model Comparison Table & Metrics Exporter."""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("model_comparison")


def build_model_comparison_table(
    eval_results: List[Dict[str, Any]],
    output_dir: Path,
    filename: str = "model_comparison.csv",
) -> pd.DataFrame:
    """Consolidates model evaluation metrics across regions, diseases, horizons, and models.

    Columns:
    --------
    REGION, DISEASE, MODEL, HORIZON, MAE, RMSE, SMAPE, MASE
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(eval_results)
    if df.empty:
        logger.warning("Empty evaluation results list. No comparison table written.")
        return df

    req_cols = ["region", "disease", "model", "horizon", "MAE", "RMSE", "SMAPE", "MASE"]
    avail_cols = [c for c in req_cols if c in df.columns]
    df = df[avail_cols].sort_values(["region", "disease", "horizon", "MAE"]).reset_index(drop=True)

    out_path = output_dir / filename
    df.to_csv(out_path, index=False)
    logger.info(f"Model comparison table saved to {out_path}")

    return df
