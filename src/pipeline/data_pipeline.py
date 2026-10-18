"""Data Pipeline Orchestrator for EpiWatch Phase 1.

Pipeline Flow:
RAW CSV -> CSV Loader -> Schema Validation -> Data Quality Check -> Cleaning ->
Temporal Alignment -> Feature Engineering -> Safe Data Joining -> Final Validation ->
Processed CSV / Feature CSV / Data Quality Reports.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

from src.features.environmental import generate_environmental_features
from src.features.epidemiological import generate_epidemiological_features
from src.features.healthcare import generate_healthcare_features
from src.features.temporal import generate_temporal_features
from src.ingestion.loader import load_csv
from src.ingestion.validator import DataValidator
from src.preprocessing.cleaner import DataCleaner
from src.utils.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger("data_pipeline")


def safe_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on_keys: list,
    how: str = "left",
    join_name: str = "join"
) -> pd.DataFrame:
    """Perform a merge while validating row counts to prevent unintended many-to-many joins.

    Raises:
        ValueError: If row count after join differs unexpectedly from left_df.
    """
    initial_rows = len(left_df)
    logger.info(f"[{join_name}] Joining on {on_keys}. Left rows before join: {initial_rows}")

    # Drop duplicate keys from right_df if any exist to guarantee 1-to-1 or N-to-1 join
    right_dedup = right_df.drop_duplicates(subset=on_keys).copy()

    # Drop overlapping columns except join keys
    overlapping_cols = [c for c in right_dedup.columns if c in left_df.columns and c not in on_keys]
    if overlapping_cols:
        logger.debug(f"[{join_name}] Dropping overlapping columns from right dataset: {overlapping_cols}")
        right_dedup.drop(columns=overlapping_cols, inplace=True)

    joined_df = pd.merge(left_df, right_dedup, on=on_keys, how=how)
    final_rows = len(joined_df)

    if final_rows != initial_rows:
        error_msg = (
            f"[{join_name}] CRITICAL JOIN ERROR: Row count changed from {initial_rows} to {final_rows}! "
            f"Many-to-many join detected. Aborting pipeline."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"[{join_name}] Join successful. Row count preserved: {final_rows}")
    return joined_df


def run_pipeline(config_path: str = "config.yaml") -> Dict[str, Path]:
    """Execute complete Phase 1 Data Pipeline.

    Returns:
        Dict mapping artifact descriptions to output file paths.
    """
    logger.info("==================================================")
    logger.info("Starting EpiWatch Phase 1 Data Engineering Pipeline")
    logger.info("==================================================")

    config = get_config(config_path)

    # 1. Setup Directories
    raw_dir = config.raw_data_dir
    processed_dir = config.processed_data_dir
    features_dir = config.features_dir
    reports_dir = config.reports_dir

    for d in [raw_dir, processed_dir, features_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Data Files Definitions
    raw_files = config.get("raw_files", {})
    cleaner = DataCleaner()
    validator = DataValidator(config.get("schemas", {}))

    loaded_datasets = {}
    validation_reports = []

    # 3. Load, Validate, Clean Raw Datasets
    # 3A. Epidemiological
    epi_path = raw_files.get("epidemiological", "Dataset/05_epidemiological_daily_bhopal.csv")
    df_epi_raw = load_csv(epi_path)
    _, r_epi = validator.validate_dataset(df_epi_raw, "epidemiological", date_column="date")
    validation_reports.append(r_epi)

    df_epi_clean, _ = cleaner.clean_dataset(
        df_epi_raw,
        dataset_type="epidemiological",
        date_col="date",
        group_cols=["region", "disease"],
        data_status="SOURCE_CALIBRATED_SYNTHETIC"
    )

    # 3B. Environmental
    env_path = raw_files.get("environmental", "Dataset/02_environmental_daily_bhopal.csv")
    df_env_raw = load_csv(env_path)
    _, r_env = validator.validate_dataset(df_env_raw, "environmental", date_column="date")
    validation_reports.append(r_env)

    df_env_clean, _ = cleaner.clean_dataset(
        df_env_raw,
        dataset_type="environmental",
        date_col="date",
        group_cols=["region"],
        data_status="SOURCE_CALIBRATED_SYNTHETIC"
    )

    # 3C. Healthcare
    hc_path = raw_files.get("healthcare", "Dataset/03_healthcare_monthly_bhopal.csv")
    df_hc_raw = load_csv(hc_path)
    _, r_hc = validator.validate_dataset(df_hc_raw, "healthcare", date_column="month")
    validation_reports.append(r_hc)

    df_hc_clean, _ = cleaner.clean_dataset(
        df_hc_raw,
        dataset_type="healthcare",
        date_col="month",
        group_cols=["region"],
        data_status="SOURCE_CALIBRATED_SYNTHETIC"
    )

    # 3D. Demographics
    demo_path = raw_files.get("demographics", "Dataset/01_demographics_bhopal.csv")
    df_demo_raw = load_csv(demo_path)
    _, r_demo = validator.validate_dataset(df_demo_raw, "demographics")
    validation_reports.append(r_demo)

    df_demo_clean, _ = cleaner.clean_dataset(
        df_demo_raw,
        dataset_type="demographics",
        group_cols=["region"],
        data_status="REAL_HISTORICAL"  # Anchor Census 2011 values
    )

    # Save Quality Reports
    validator.save_quality_reports(validation_reports, reports_dir)

    # 4. Save Processed Cleaned Datasets
    epi_clean_file = processed_dir / "epidemiological_clean.csv"
    env_clean_file = processed_dir / "environmental_clean.csv"
    hc_clean_file = processed_dir / "healthcare_clean.csv"
    demo_clean_file = processed_dir / "demographics_clean.csv"

    df_epi_clean.to_csv(epi_clean_file, index=False)
    df_env_clean.to_csv(env_clean_file, index=False)
    df_hc_clean.to_csv(hc_clean_file, index=False)
    df_demo_clean.to_csv(demo_clean_file, index=False)
    logger.info("Saved all cleaned processed datasets to data/processed/")

    # 5. Extract Population Anchor
    bhopal_pop = 2368145.0  # Bhopal Census 2011 population anchor
    if "population_2011" in df_demo_clean.columns:
        bhopal_pop = float(df_demo_clean["population_2011"].iloc[0])

    # 6. Feature Engineering per Category
    # 6A. Epidemiological Features
    df_epi_feat = generate_epidemiological_features(
        df_epi_clean, population=bhopal_pop, cases_col="new_cases", group_cols=["region", "disease"]
    )
    df_epi_feat = generate_temporal_features(df_epi_feat, date_col="date")

    # 6B. Environmental Features
    df_env_feat = generate_environmental_features(df_env_clean, group_cols=["region"])
    df_env_feat = generate_temporal_features(df_env_feat, date_col="date")

    # 6C. Healthcare Features
    df_hc_feat = generate_healthcare_features(df_hc_clean, population=bhopal_pop)
    df_hc_feat = generate_temporal_features(df_hc_feat, date_col="month")

    # Save Feature Datasets
    epi_feat_file = features_dir / "epidemiological_features.csv"
    env_feat_file = features_dir / "environmental_features.csv"
    hc_feat_file = features_dir / "healthcare_features.csv"

    df_epi_feat.to_csv(epi_feat_file, index=False)
    df_env_feat.to_csv(env_feat_file, index=False)
    df_hc_feat.to_csv(hc_feat_file, index=False)
    logger.info("Saved category feature datasets to data/features/")

    # 7. Safe Joining -> EpiWatch Master Feature Dataset
    logger.info("Assembling EpiWatch Master Feature Dataset...")
    master_df = df_epi_feat.copy()

    # Step A: Join Environmental Daily Data on ['date', 'region']
    if "date" in master_df.columns and "date" in df_env_feat.columns:
        master_df["date"] = pd.to_datetime(master_df["date"])
        df_env_feat["date"] = pd.to_datetime(df_env_feat["date"])
        master_df = safe_join(master_df, df_env_feat, on_keys=["date", "region"], join_name="Epi + Env")

    # Step B: Join Demographics Static Data on ['region']
    if "region" in master_df.columns and "region" in df_demo_clean.columns:
        master_df = safe_join(master_df, df_demo_clean, on_keys=["region"], join_name="EpiEnv + Demo")

    # Step C: Join Healthcare Monthly Data on ['region', 'year_month']
    date_col = "date" if "date" in master_df.columns else "month"
    hc_date_col = "month" if "month" in df_hc_feat.columns else "date"

    if date_col in master_df.columns and hc_date_col in df_hc_feat.columns:
        master_df["year_month"] = pd.to_datetime(master_df[date_col]).dt.to_period("M").astype(str)
        df_hc_feat_copy = df_hc_feat.copy()
        df_hc_feat_copy["year_month"] = pd.to_datetime(df_hc_feat_copy[hc_date_col]).dt.to_period("M").astype(str)

        master_df = safe_join(
            master_df, df_hc_feat_copy, on_keys=["region", "year_month"], join_name="EpiEnvDemo + Healthcare"
        )
        master_df.drop(columns=["year_month"], inplace=True, errors="ignore")

    # 8. Save Master Feature Dataset
    master_file = features_dir / "epiwatch_master_features.csv"
    master_df.to_csv(master_file, index=False)
    logger.info(f"Successfully generated Master Feature Dataset ({len(master_df)} rows, {len(master_df.columns)} cols): {master_file}")

    logger.info("==================================================")
    logger.info("EpiWatch Phase 1 Pipeline Completed Successfully!")
    logger.info("==================================================")

    return {
        "epidemiological_clean": epi_clean_file,
        "environmental_clean": env_clean_file,
        "healthcare_clean": hc_clean_file,
        "demographics_clean": demo_clean_file,
        "epidemiological_features": epi_feat_file,
        "environmental_features": env_feat_file,
        "healthcare_features": hc_feat_file,
        "master_features": master_file,
        "data_quality_summary": reports_dir / "data_quality_summary.csv"
    }


if __name__ == "__main__":
    run_pipeline()
