"""
test_data_loader.py — Tests for DunnhumbyDataLoader
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


# Minimal config for testing
@pytest.fixture
def test_config(tmp_path):
    """Create minimal config with temp directories."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    config = {
        "project": {"name": "test_clv", "version": "1.0.0", "random_seed": 42},
        "data": {
            "raw_dir": str(raw_dir),
            "interim_dir": str(tmp_path / "data" / "interim"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "files": {
                "transaction_data": str(raw_dir / "transaction_data.csv"),
                "hh_demographic": str(raw_dir / "hh_demographic.csv"),
                "campaign_table": str(raw_dir / "campaign_table.csv"),
                "campaign_desc": str(raw_dir / "campaign_desc.csv"),
                "coupon": str(raw_dir / "coupon.csv"),
                "coupon_redempt": str(raw_dir / "coupon_redempt.csv"),
                "product": str(raw_dir / "product.csv"),
                "causal_data": str(raw_dir / "causal_data.csv"),
            },
        },
    }
    return config


@pytest.fixture
def sample_transactions(test_config):
    """Create a small sample transaction CSV for testing."""
    df = pd.DataFrame({
        "household_key": [1, 1, 2, 2, 3],
        "BASKET_ID": [100, 101, 200, 201, 300],
        "DAY": [1, 10, 5, 15, 20],
        "PRODUCT_ID": [1001, 1002, 1001, 1003, 1002],
        "QUANTITY": [2, 1, 3, 1, 2],
        "SALES_VALUE": [5.99, 3.49, 8.99, 2.99, 4.99],
        "STORE_ID": [10, 10, 20, 20, 10],
        "RETAIL_DISC": [-0.50, 0.0, -1.00, 0.0, -0.25],
        "TRANS_TIME": [1200, 1400, 900, 1100, 1600],
        "WEEK_NO": [1, 5, 3, 10, 15],
        "COUPON_DISC": [0.0, -0.25, 0.0, 0.0, -0.50],
        "COUPON_MATCH_DISC": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    path = test_config["data"]["files"]["transaction_data"]
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_demographics(test_config):
    """Create sample demographics CSV."""
    df = pd.DataFrame({
        "household_key": [1, 2],
        "AGE_DESC": ["25-34", "45-54"],
        "MARITAL_STATUS_CODE": ["A", "B"],
        "INCOME_DESC": ["50-74K", "100-124K"],
        "HOMEOWNER_DESC": ["Homeowner", "Renter"],
        "HH_COMP_DESC": ["2 Adults Kids", "1 Adult Kids"],
        "HOUSEHOLD_SIZE_DESC": ["3", "2"],
        "KID_CATEGORY_DESC": ["1", "1"],
    })
    path = test_config["data"]["files"]["hh_demographic"]
    df.to_csv(path, index=False)
    return path


class TestDunnhumbyDataLoader:
    """Tests for DunnhumbyDataLoader."""

    def test_init_creates_file_paths(self, test_config):
        """Test that loader initializes with correct file paths."""
        from src.data.data_loader import DunnhumbyDataLoader
        loader = DunnhumbyDataLoader(test_config)
        assert hasattr(loader, "file_paths")
        assert "transaction_data" in loader.file_paths

    def test_load_transactions_returns_dataframe(
        self, test_config, sample_transactions
    ):
        """Test transaction loading returns a DataFrame with expected shape."""
        from src.data.data_loader import DunnhumbyDataLoader
        loader = DunnhumbyDataLoader(test_config)
        df = loader.load_transactions()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "household_key" in df.columns
        assert "SALES_VALUE" in df.columns

    def test_load_transactions_dtype_optimization(
        self, test_config, sample_transactions
    ):
        """Test that dtypes are downcast for memory optimization."""
        from src.data.data_loader import DunnhumbyDataLoader
        loader = DunnhumbyDataLoader(test_config)
        df = loader.load_transactions()
        # household_key should be int32 (not default int64)
        assert df["household_key"].dtype == np.int32

    def test_load_demographics_returns_dataframe(
        self, test_config, sample_demographics
    ):
        """Test demographics loading."""
        from src.data.data_loader import DunnhumbyDataLoader
        loader = DunnhumbyDataLoader(test_config)
        df = loader.load_demographics()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "household_key" in df.columns

    def test_explore_returns_summary(self, test_config, sample_transactions):
        """Test explore method returns expected summary dict."""
        from src.data.data_loader import DunnhumbyDataLoader
        loader = DunnhumbyDataLoader(test_config)
        df = loader.load_transactions()
        summary = loader.explore(df, "test_table")
        assert "shape" in summary
        assert "memory_mb" in summary
