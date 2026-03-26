"""
make_dataset.py
===============
Data ingestion pipeline: loads raw data from source, validates schema,
and persists cleaned data to the appropriate data directories.

Usage:
    python -m src.data.make_dataset configs/config.yaml
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration Loader
# ==============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded from %s", config_path)
    return config


# ==============================================================================
# Data Loading
# ==============================================================================

def load_raw_data(config: dict) -> pd.DataFrame:
    """Load raw data from the configured source path.

    Supports CSV files. Extend this function to add API or database ingestion.

    Args:
        config: Project configuration dictionary.

    Returns:
        Raw DataFrame with parsed datetime index.
    """
    raw_path = Path(config["data"]["raw_file"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_path}. "
            "Please place your data in the 'data/raw/' directory."
        )

    datetime_col = config["data"]["columns"]["datetime"]
    date_format = config["data"].get("date_format")

    df = pd.read_csv(
        raw_path,
        parse_dates=[datetime_col],
        date_format=date_format,
    )

    logger.info("Loaded raw data: %d rows, %d columns", *df.shape)
    return df


# ==============================================================================
# Validation
# ==============================================================================

def validate_schema(df: pd.DataFrame, config: dict) -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: Input DataFrame.
        config: Project configuration dictionary.

    Raises:
        ValueError: If expected columns are missing.
    """
    expected_cols = list(config["data"]["columns"].values())
    missing = [c for c in expected_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    logger.info("Schema validation passed.")


# ==============================================================================
# Data Saving
# ==============================================================================

def save_processed_data(df: pd.DataFrame, config: dict) -> Path:
    """Save the validated dataset to the processed data directory.

    Args:
        df: Validated DataFrame.
        config: Project configuration dictionary.

    Returns:
        Path to the saved file.
    """
    output_path = Path(config["data"]["processed_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Processed data saved to %s", output_path)
    return output_path


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main(config_path: str) -> None:
    """Run the full data ingestion pipeline.

    Steps:
        1. Load configuration
        2. Load raw data from CSV
        3. Validate schema (required columns)
        4. Save validated data to processed directory

    Args:
        config_path: Path to YAML configuration file.
    """
    config = load_config(config_path)

    # --- Setup logging ---
    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("STAGE 1: Data Ingestion")
    logger.info("=" * 60)

    # 1. Load
    df = load_raw_data(config)

    # 2. Validate
    validate_schema(df, config)

    # 3. Save
    save_processed_data(df, config)

    logger.info("Data ingestion pipeline completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.data.make_dataset <config_path>")
        sys.exit(1)
    main(sys.argv[1])
