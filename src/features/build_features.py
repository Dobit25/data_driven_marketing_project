"""
build_features.py
=================
Preprocessing pipeline: handles missing values, outlier treatment,
stationarity testing (ADF, KPSS), ACF/PACF analysis, and feature
engineering for time series modeling.

Usage:
    python -m src.features.build_features configs/config.yaml
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================================
# Missing Value Handling
# ==============================================================================

def handle_missing_values(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Handle missing values based on the configured strategy.

    Supported strategies: 'ffill', 'bfill', 'interpolate', 'drop'.

    Args:
        df: Input DataFrame.
        config: Project configuration dictionary.

    Returns:
        DataFrame with missing values handled.
    """
    strategy = config["preprocessing"]["missing_strategy"]
    initial_nulls = df.isnull().sum().sum()

    if strategy == "ffill":
        df = df.ffill()
    elif strategy == "bfill":
        df = df.bfill()
    elif strategy == "interpolate":
        df = df.interpolate(method="time")
    elif strategy == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")

    remaining_nulls = df.isnull().sum().sum()
    logger.info(
        "Missing values: %d -> %d (strategy: %s)",
        initial_nulls, remaining_nulls, strategy,
    )
    return df


# ==============================================================================
# Outlier Detection & Removal
# ==============================================================================

def remove_outliers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remove outliers using the IQR method.

    Args:
        df: Input DataFrame.
        config: Project configuration dictionary.

    Returns:
        DataFrame with outliers removed.
    """
    multiplier = config["preprocessing"].get("outlier_iqr_multiplier")
    if multiplier is None:
        logger.info("Outlier removal disabled.")
        return df

    target_col = config["data"]["columns"]["target"]
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    before = len(df)
    df = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
    after = len(df)

    logger.info(
        "Outlier removal (IQR x%.1f): %d -> %d rows",
        multiplier, before, after,
    )
    return df


# ==============================================================================
# Stationarity Tests
# ==============================================================================

def test_stationarity(series: pd.Series, config: dict) -> dict:
    """Run Augmented Dickey-Fuller and KPSS stationarity tests.

    Args:
        series: Time series to test.
        config: Project configuration dictionary.

    Returns:
        Dictionary with test results.
    """
    results = {}
    alpha_adf = config["statistical_tests"]["adf_significance"]
    alpha_kpss = config["statistical_tests"]["kpss_significance"]

    # --- ADF Test ---
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(
        series.dropna(), autolag="AIC"
    )
    adf_stationary = adf_p < alpha_adf
    results["adf"] = {
        "statistic": adf_stat,
        "p_value": adf_p,
        "lags_used": adf_lags,
        "is_stationary": adf_stationary,
    }
    logger.info(
        "ADF Test — Statistic: %.4f, p-value: %.4f => %s",
        adf_stat, adf_p,
        "Stationary" if adf_stationary else "Non-Stationary",
    )

    # --- KPSS Test ---
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
        series.dropna(), regression="c", nlags="auto"
    )
    kpss_stationary = kpss_p > alpha_kpss
    results["kpss"] = {
        "statistic": kpss_stat,
        "p_value": kpss_p,
        "lags_used": kpss_lags,
        "is_stationary": kpss_stationary,
    }
    logger.info(
        "KPSS Test — Statistic: %.4f, p-value: %.4f => %s",
        kpss_stat, kpss_p,
        "Stationary" if kpss_stationary else "Non-Stationary",
    )

    return results


# ==============================================================================
# ACF / PACF Analysis
# ==============================================================================

def compute_acf_pacf(series: pd.Series, config: dict) -> dict:
    """Compute ACF and PACF values for lag analysis.

    Args:
        series: Time series data.
        config: Project configuration dictionary.

    Returns:
        Dictionary with ACF and PACF arrays.
    """
    nlags_acf = config["statistical_tests"]["acf_lags"]
    nlags_pacf = config["statistical_tests"]["pacf_lags"]

    acf_values = acf(series.dropna(), nlags=nlags_acf)
    pacf_values = pacf(series.dropna(), nlags=nlags_pacf)

    logger.info("ACF/PACF computed with %d / %d lags.", nlags_acf, nlags_pacf)
    return {"acf": acf_values, "pacf": pacf_values}


# ==============================================================================
# Feature Engineering
# ==============================================================================

def build_time_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Extract time-based features from the datetime column.

    Creates: hour, day_of_week, day_of_month, month, is_weekend.

    Args:
        df: Input DataFrame with a datetime column.
        config: Project configuration dictionary.

    Returns:
        DataFrame with additional time features.
    """
    dt_col = config["data"]["columns"]["datetime"]
    df[dt_col] = pd.to_datetime(df[dt_col])

    df["hour"] = df[dt_col].dt.hour
    df["day_of_week"] = df[dt_col].dt.dayofweek
    df["day_of_month"] = df[dt_col].dt.day
    df["month"] = df[dt_col].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    logger.info("Time-based features created: hour, day_of_week, day_of_month, month, is_weekend")
    return df


# ==============================================================================
# Train/Test Split
# ==============================================================================

def split_data(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets based on temporal ordering.

    Args:
        df: Input DataFrame.
        config: Project configuration dictionary.

    Returns:
        Tuple of (train_df, test_df).
    """
    ratio = config["preprocessing"]["train_ratio"]
    split_idx = int(len(df) * ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    logger.info("Train/test split: %d / %d rows (ratio=%.2f)", len(train), len(test), ratio)
    return train, test


# ==============================================================================
# Main Pipeline
# ==============================================================================

def main(config_path: str) -> None:
    """Run the full feature engineering pipeline.

    Steps:
        1. Load processed data
        2. Handle missing values
        3. Remove outliers
        4. Run stationarity tests (ADF, KPSS)
        5. Compute ACF/PACF
        6. Build time features
        7. Split into train/test
        8. Save outputs

    Args:
        config_path: Path to YAML configuration file.
    """
    config = load_config(config_path)

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info("=" * 60)
    logger.info("STAGE 2: Feature Engineering & Statistical Testing")
    logger.info("=" * 60)

    # 1. Load data
    processed_path = Path(config["data"]["processed_file"])
    df = pd.read_csv(processed_path, parse_dates=[config["data"]["columns"]["datetime"]])
    logger.info("Loaded processed data: %d rows", len(df))

    # 2. Handle missing values
    target_col = config["data"]["columns"]["target"]
    df = handle_missing_values(df, config)

    # 3. Remove outliers
    df = remove_outliers(df, config)

    # 4. Stationarity tests
    stationarity = test_stationarity(df[target_col], config)

    # 5. ACF/PACF
    acf_pacf = compute_acf_pacf(df[target_col], config)

    # 6. Build time features
    df = build_time_features(df, config)

    # 7. Train/test split
    train_df, test_df = split_data(df, config)

    # 8. Save
    interim_dir = Path(config["data"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(interim_dir / "train.csv", index=False)
    test_df.to_csv(interim_dir / "test.csv", index=False)

    logger.info("Feature engineering pipeline completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.features.build_features <config_path>")
        sys.exit(1)
    main(sys.argv[1])
