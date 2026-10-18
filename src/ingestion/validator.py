"""Data Validation and Quality Reporting for EpiWatch platform."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("validator")


class DataValidator:
    """Validates datasets against defined schemas, data types, missingness,

    duplicates, and domain range constraints.
    """

    def __init__(self, schema_config: Dict[str, Any]):
        """Initialize validator with schema dictionary from configuration.

        Args:
            schema_config: Schema definitions dict for datasets.
        """
        self.schema_config = schema_config

    def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        date_column: str = "date"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Perform comprehensive validation on a dataset DataFrame.

        Returns:
            Tuple of (validation_passed_bool, report_dict)
        """
        logger.info(f"Starting validation for dataset: '{dataset_name}'")
        report: Dict[str, Any] = {
            "dataset": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "validation_status": "PASSED",
            "schema_issues": [],
            "data_type_issues": [],
            "missing_values": {},
            "total_missing_cells": 0,
            "duplicate_rows": 0,
            "invalid_value_issues": [],
            "invalid_rows_count": 0,
            "date_range": None,
            "unique_regions": 0,
            "unique_diseases": 0,
            "numerical_summary": {}
        }

        dataset_schema = self.schema_config.get(dataset_name, {})
        required_cols = dataset_schema.get("required_columns", [])
        numeric_cols = dataset_schema.get("numeric_columns", [])
        range_checks = dataset_schema.get("range_checks", {})
        non_negative_cols = dataset_schema.get("non_negative_columns", [])
        key_cols = dataset_schema.get("key_columns", [])

        # 1. Schema Validation
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            issue = f"Missing required columns: {missing_required}"
            logger.error(f"[{dataset_name}] {issue}")
            report["schema_issues"].append(issue)
            report["validation_status"] = "FAILED"

        unexpected_cols = [c for c in df.columns if c not in required_cols and c not in numeric_cols]
        if unexpected_cols:
            warn = f"Unexpected columns found: {unexpected_cols}"
            logger.warning(f"[{dataset_name}] {warn}")
            report["schema_issues"].append(f"WARNING: {warn}")

        # 2. Data Type & Date Validation
        date_col = date_column if date_column in df.columns else ("month" if "month" in df.columns else None)
        if date_col:
            try:
                parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
                invalid_dates_count = int(parsed_dates.isna().sum() - df[date_col].isna().sum())
                if invalid_dates_count > 0:
                    issue = f"Found {invalid_dates_count} unparseable date values in column '{date_col}'"
                    logger.error(f"[{dataset_name}] {issue}")
                    report["data_type_issues"].append(issue)
                    report["validation_status"] = "FAILED"
                else:
                    valid_dates = parsed_dates.dropna()
                    if not valid_dates.empty:
                        report["date_range"] = {
                            "min_date": str(valid_dates.min().strftime("%Y-%m-%d")),
                            "max_date": str(valid_dates.max().strftime("%Y-%m-%d"))
                        }
            except Exception as e:
                report["data_type_issues"].append(f"Error parsing dates in '{date_col}': {str(e)}")

        for col in numeric_cols:
            if col in df.columns:
                non_numeric_count = pd.to_numeric(df[col], errors="coerce").isna().sum() - df[col].isna().sum()
                if non_numeric_count > 0:
                    issue = f"Column '{col}' has {non_numeric_count} non-numeric values"
                    logger.error(f"[{dataset_name}] {issue}")
                    report["data_type_issues"].append(issue)
                    report["validation_status"] = "FAILED"

        if "region" in df.columns:
            report["unique_regions"] = int(df["region"].nunique(dropna=True))
        if "disease" in df.columns:
            report["unique_diseases"] = int(df["disease"].nunique(dropna=True))

        # 3. Missing Values Reporting
        missing_by_col = df.isna().sum().to_dict()
        total_missing = int(df.isna().sum().sum())
        report["total_missing_cells"] = total_missing
        report["missing_values"] = {
            col: {
                "count": int(count),
                "percentage": round(float(count / len(df) * 100), 2) if len(df) > 0 else 0.0
            }
            for col, count in missing_by_col.items() if count > 0
        }
        if total_missing > 0:
            logger.warning(f"[{dataset_name}] Detected {total_missing} missing cells across affected columns.")

        # 4. Duplicate Records Detection
        valid_keys = [k for k in key_cols if k in df.columns]
        if valid_keys:
            duplicates = int(df.duplicated(subset=valid_keys, keep=False).sum())
            report["duplicate_rows"] = duplicates
            if duplicates > 0:
                logger.warning(f"[{dataset_name}] Detected {duplicates} duplicate rows based on keys {valid_keys}")
        else:
            duplicates = int(df.duplicated(keep=False).sum())
            report["duplicate_rows"] = duplicates

        # 5. Invalid Numerical Values & Range Checks
        invalid_row_indices = set()

        # Check negative values where forbidden
        for col in non_negative_cols:
            if col in df.columns:
                num_series = pd.to_numeric(df[col], errors="coerce")
                neg_mask = num_series < 0
                neg_count = int(neg_mask.sum())
                if neg_count > 0:
                    issue = f"Column '{col}' contains {neg_count} negative values"
                    logger.error(f"[{dataset_name}] {issue}")
                    report["invalid_value_issues"].append(issue)
                    invalid_row_indices.update(df[neg_mask].index.tolist())
                    if report["validation_status"] != "FAILED":
                        report["validation_status"] = "PASSED_WITH_WARNINGS"

        # Range checks (e.g. humidity 0-100, oxygen 0-100)
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                num_series = pd.to_numeric(df[col], errors="coerce")
                out_of_bounds = (num_series < min_val) | (num_series > max_val)
                oob_count = int(out_of_bounds.sum())
                if oob_count > 0:
                    issue = f"Column '{col}' has {oob_count} values outside expected range [{min_val}, {max_val}]"
                    logger.warning(f"[{dataset_name}] {issue}")
                    report["invalid_value_issues"].append(issue)
                    invalid_row_indices.update(df[out_of_bounds].index.tolist())
                    if report["validation_status"] == "PASSED":
                        report["validation_status"] = "PASSED_WITH_WARNINGS"

        report["invalid_rows_count"] = len(invalid_row_indices)

        # 6. Min/Max Summary for Numeric Columns
        for col in df.select_dtypes(include=[np.number]).columns:
            report["numerical_summary"][col] = {
                "min": round(float(df[col].min()), 4) if not df[col].dropna().empty else None,
                "max": round(float(df[col].max()), 4) if not df[col].dropna().empty else None,
                "mean": round(float(df[col].mean()), 4) if not df[col].dropna().empty else None,
            }

        logger.info(f"[{dataset_name}] Validation finished with status: {report['validation_status']}")
        passed_bool = report["validation_status"] in ["PASSED", "PASSED_WITH_WARNINGS"]
        return passed_bool, report

    def save_quality_reports(
        self,
        reports: List[Dict[str, Any]],
        output_dir: Path
    ) -> Path:
        """Save individual JSON quality reports and summary CSV table.

        Args:
            reports: List of validation report dictionaries.
            output_dir: Directory where reports will be saved.

        Returns:
            Path to summary CSV file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        for r in reports:
            ds_name = r["dataset"]
            json_file = output_dir / f"{ds_name}_quality_report.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2)
            logger.info(f"Saved machine-readable quality report: {json_file}")

            summary_rows.append({
                "dataset": ds_name,
                "rows": r["rows"],
                "columns": r["columns"],
                "missing_cells": r["total_missing_cells"],
                "duplicates": r["duplicate_rows"],
                "invalid_rows": r["invalid_rows_count"],
                "status": r["validation_status"]
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = output_dir / "data_quality_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved data quality summary CSV: {summary_csv}")
        return summary_csv
