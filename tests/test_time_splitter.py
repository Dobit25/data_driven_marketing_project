"""
test_time_splitter.py — Tests for TimeSplitter
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def splitter_config():
    """Minimal config for TimeSplitter."""
    return {
        "splitting": {
            "calibration_end_week": 75,
            "total_weeks": 102,
        },
    }


@pytest.fixture
def mock_transactions():
    """Create mock transactions spanning weeks 1-102."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "household_key": np.random.choice(range(1, 51), n),
        "BASKET_ID": np.arange(1000, 1000 + n),
        "WEEK_NO": np.random.randint(1, 103, n),
        "PRODUCT_ID": np.random.randint(1, 100, n),
        "STORE_ID": np.random.choice([10, 20, 30], n),
        "QUANTITY": np.random.randint(1, 10, n),
        "SALES_VALUE": np.random.uniform(1.0, 50.0, n).round(2),
        "RETAIL_DISC": np.zeros(n),
        "COUPON_DISC": np.zeros(n),
        "COUPON_MATCH_DISC": np.zeros(n),
    })


class TestTimeSplitter:
    """Tests for TimeSplitter."""

    def test_split_returns_two_dataframes(self, splitter_config, mock_transactions):
        """Test that split returns calibration and holdout DataFrames."""
        from src.features.time_splitter import TimeSplitter
        splitter = TimeSplitter(splitter_config)
        cal, holdout = splitter.split(mock_transactions)
        assert isinstance(cal, pd.DataFrame)
        assert isinstance(holdout, pd.DataFrame)
        assert len(cal) > 0
        assert len(holdout) > 0

    def test_no_temporal_overlap(self, splitter_config, mock_transactions):
        """Test that calibration and holdout periods do not overlap."""
        from src.features.time_splitter import TimeSplitter
        splitter = TimeSplitter(splitter_config)
        cal, holdout = splitter.split(mock_transactions)

        cal_max_week = cal["WEEK_NO"].max()
        holdout_min_week = holdout["WEEK_NO"].min()
        assert cal_max_week < holdout_min_week, (
            f"Temporal overlap detected: cal max={cal_max_week}, "
            f"holdout min={holdout_min_week}"
        )

    def test_calibration_within_bounds(self, splitter_config, mock_transactions):
        """Test calibration data only contains weeks <= cutoff."""
        from src.features.time_splitter import TimeSplitter
        splitter = TimeSplitter(splitter_config)
        cal, _ = splitter.split(mock_transactions)
        cutoff = splitter_config["splitting"]["calibration_end_week"]
        assert cal["WEEK_NO"].max() <= cutoff

    def test_holdout_within_bounds(self, splitter_config, mock_transactions):
        """Test holdout data only contains weeks > cutoff."""
        from src.features.time_splitter import TimeSplitter
        splitter = TimeSplitter(splitter_config)
        _, holdout = splitter.split(mock_transactions)
        cutoff = splitter_config["splitting"]["calibration_end_week"]
        assert holdout["WEEK_NO"].min() > cutoff

    def test_no_data_loss(self, splitter_config, mock_transactions):
        """Test that no rows are lost during split."""
        from src.features.time_splitter import TimeSplitter
        splitter = TimeSplitter(splitter_config)
        cal, holdout = splitter.split(mock_transactions)
        assert len(cal) + len(holdout) == len(mock_transactions)

    def test_get_split_info(self, splitter_config, mock_transactions):
        """Test split info method returns expected keys."""
        from src.features.time_splitter import TimeSplitter
        splitter = TimeSplitter(splitter_config)
        splitter.split(mock_transactions)
        info = splitter.get_split_info()
        assert isinstance(info, dict)
