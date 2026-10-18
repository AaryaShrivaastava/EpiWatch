"""Domain-aware missing value handling for EpiWatch datasets."""

from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("missing_values")


class MissingValueHandler:
    """Handles missing values in epidemiological, environmental, healthcare,

    and demographic datasets using domain-aware strategies.
    """

    def __init__(self):
        self.imputation_logs: List[Dict[str, str]] = []

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        dataset_type: str,
        group_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply dataset-specific missing value imputation strategies.

        Strategies:
        - Epidemiological time-series: linear interpolation for small numeric gaps per region/disease.
        - Environmental time-series: linear interpolation for numeric weather variables per region.
        - Healthcare monthly: forward-fill within region for slow-moving capacity indicators.
        - Demographics static: forward-fill within region for static population figures.
        - Retain NaN when imputation would create false information.

        Returns:
            Tuple of (imputed_df, imputation_summary_df)
        """
        df_clean = df.copy()
        summary_list = []

        for col in df_clean.columns:
            missing_before = int(df_clean[col].isna().sum())
            if missing_before == 0:
                continue

            method_used = "retained_nan"

            # 1. Static demographic values
            if dataset_type == "demographics":
                if group_cols and "region" in group_cols:
                    df_clean[col] = df_clean.groupby("region")[col].ffill().bfill()
                else:
                    df_clean[col] = df_clean[col].ffill().bfill()
                method_used = "forward_backward_fill_region"

            # 2. Time series numeric variables (epidemiological / environmental)
            elif dataset_type in ["epidemiological", "environmental"]:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    if group_cols:
                        df_clean[col] = df_clean.groupby(group_cols)[col].transform(
                            lambda g: g.interpolate(method="linear", limit=3)
                        )
                    else:
                        df_clean[col] = df_clean[col].interpolate(method="linear", limit=3)
                    method_used = "linear_interpolation_limit_3"

            # 3. Monthly healthcare indicators
            elif dataset_type == "healthcare":
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    if group_cols and "region" in group_cols:
                        df_clean[col] = df_clean.groupby("region")[col].ffill().bfill()
                    else:
                        df_clean[col] = df_clean[col].ffill().bfill()
                    method_used = "forward_backward_fill_region"

            missing_after = int(df_clean[col].isna().sum())
            imputed_count = missing_before - missing_after

            if imputed_count > 0:
                logger.info(
                    f"Imputed {imputed_count} missing values in '{col}' using method '{method_used}'"
                )

            summary_list.append({
                "column": col,
                "missing_before": missing_before,
                "missing_after": missing_after,
                "method_used": method_used
            })

        summary_df = pd.DataFrame(summary_list)
        return df_clean, summary_df
