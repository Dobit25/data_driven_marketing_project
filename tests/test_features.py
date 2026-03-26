"""
test_features.py
================
Unit tests for feature engineering module (src.features.build_features).
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    build_time_features,
    handle_missing_values,
    remove_outliers,
    split_data,
)


@pytest.fixture
def sample_config():
    """Minimal config for testing."""
    return {
        "data": {
            "columns": {"datetime": "datetime", "target": "concurrent_players"},
        },
        "preprocessing": {
            "missing_strategy": "interpolate",
            "outlier_iqr_multiplier": 1.5,
            "train_ratio": 0.8,
        },
    }


@pytest.fixture
def sample_df():
    """Sample DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame({
        "datetime": dates,
        "concurrent_players": np.random.randint(500, 5000, size=100),
    })


class TestHandleMissingValues:
    """Tests for missing value handling."""

    def test_interpolate_fills_nulls(self, sample_df, sample_config):
        sample_df.loc[5, "concurrent_players"] = np.nan
        result = handle_missing_values(sample_df, sample_config)
        assert result["concurrent_players"].isnull().sum() == 0

    def test_drop_removes_rows(self, sample_df, sample_config):
        sample_config["preprocessing"]["missing_strategy"] = "drop"
        sample_df.loc[5, "concurrent_players"] = np.nan
        result = handle_missing_values(sample_df, sample_config)
        assert len(result) == 99


class TestRemoveOutliers:
    """Tests for outlier removal."""

    def test_removes_extreme_values(self, sample_df, sample_config):
        sample_df.loc[0, "concurrent_players"] = 999999
        result = remove_outliers(sample_df, sample_config)
        assert len(result) < len(sample_df)

    def test_disabled_when_null(self, sample_df, sample_config):
        sample_config["preprocessing"]["outlier_iqr_multiplier"] = None
        result = remove_outliers(sample_df, sample_config)
        assert len(result) == len(sample_df)


class TestBuildTimeFeatures:
    """Tests for time feature extraction."""

    def test_creates_expected_columns(self, sample_df, sample_config):
        result = build_time_features(sample_df, sample_config)
        for col in ["hour", "day_of_week", "day_of_month", "month", "is_weekend"]:
            assert col in result.columns


class TestSplitData:
    """Tests for train/test splitting."""

    def test_correct_split_ratio(self, sample_df, sample_config):
        train, test = split_data(sample_df, sample_config)
        assert len(train) == 80
        assert len(test) == 20
