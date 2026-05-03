"""
test_rfm_builder.py — Tests for RFMBuilder
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def rfm_config():
    """Minimal config for RFMBuilder."""
    return {
        "rfm": {"entity_key": "household_key"},
        "splitting": {"calibration_end_week": 75, "total_weeks": 102},
    }


@pytest.fixture
def mock_transactions():
    """Create mock transaction data for RFM testing."""
    np.random.seed(42)
    n = 200
    week_nos = np.random.randint(1, 76, n)
    return pd.DataFrame({
        "household_key": np.random.choice([1, 2, 3, 4, 5], n),
        "BASKET_ID": np.arange(1000, 1000 + n),
        "DAY": week_nos * 7 - np.random.randint(0, 7, n),  # approx day from week
        "WEEK_NO": week_nos,
        "PRODUCT_ID": np.random.randint(1, 50, n),
        "STORE_ID": np.random.choice([10, 20, 30], n),
        "QUANTITY": np.random.randint(1, 10, n),
        "SALES_VALUE": np.random.uniform(1.0, 50.0, n).round(2),
        "RETAIL_DISC": -np.random.uniform(0, 2.0, n).round(2),
        "COUPON_DISC": np.zeros(n),
        "COUPON_MATCH_DISC": np.zeros(n),
    })


class TestRFMBuilder:
    """Tests for RFMBuilder."""

    def test_compute_rfm_returns_dataframe(self, rfm_config, mock_transactions):
        """Test that compute_rfm returns proper DataFrame."""
        from src.features.rfm_builder import RFMBuilder
        builder = RFMBuilder(rfm_config)
        rfm = builder.compute_rfm(mock_transactions, analysis_end_day=525)
        assert isinstance(rfm, pd.DataFrame)
        assert len(rfm) > 0

    def test_entity_resolution_one_row_per_household(
        self, rfm_config, mock_transactions
    ):
        """Test Constraint 1: exactly one row per household_key."""
        from src.features.rfm_builder import RFMBuilder
        builder = RFMBuilder(rfm_config)
        rfm = builder.compute_rfm(mock_transactions, analysis_end_day=525)
        assert rfm["household_key"].is_unique

    def test_rfm_columns_present(self, rfm_config, mock_transactions):
        """Test that all expected RFM columns are present."""
        from src.features.rfm_builder import RFMBuilder
        builder = RFMBuilder(rfm_config)
        rfm = builder.compute_rfm(mock_transactions, analysis_end_day=525)
        expected_cols = [
            "household_key", "Recency", "Frequency", "T",
            "Gross_Sales", "Net_Sales", "avg_monetary",
        ]
        for col in expected_cols:
            assert col in rfm.columns, f"Missing column: {col}"

    def test_frequency_is_repeat_purchases(self, rfm_config, mock_transactions):
        """Test that Frequency = total_baskets - 1 (BG/NBD convention)."""
        from src.features.rfm_builder import RFMBuilder
        builder = RFMBuilder(rfm_config)
        rfm = builder.compute_rfm(mock_transactions, analysis_end_day=525)
        assert (rfm["Frequency"] >= 0).all()
        assert (rfm["Frequency"] == rfm["total_baskets"] - 1).all()

    def test_recency_non_negative(self, rfm_config, mock_transactions):
        """Test that Recency values are non-negative."""
        from src.features.rfm_builder import RFMBuilder
        builder = RFMBuilder(rfm_config)
        rfm = builder.compute_rfm(mock_transactions, analysis_end_day=525)
        assert (rfm["Recency"] >= 0).all()

    def test_avg_monetary_zero_for_single_purchase(self, rfm_config):
        """Test avg_monetary=0 for customers with only 1 purchase."""
        from src.features.rfm_builder import RFMBuilder
        # Customer with exactly 1 basket
        txn = pd.DataFrame({
            "household_key": [99],
            "BASKET_ID": [5000],
            "DAY": [70],
            "WEEK_NO": [10],
            "PRODUCT_ID": [1],
            "STORE_ID": [10],
            "QUANTITY": [1],
            "SALES_VALUE": [10.0],
            "RETAIL_DISC": [0.0],
            "COUPON_DISC": [0.0],
            "COUPON_MATCH_DISC": [0.0],
        })
        builder = RFMBuilder(rfm_config)
        rfm = builder.compute_rfm(txn, analysis_end_day=525)
        assert rfm.loc[0, "Frequency"] == 0
        assert rfm.loc[0, "avg_monetary"] == 0

    def test_compute_rfm_summary(self, rfm_config, mock_transactions):
        """Test summary statistics computation."""
        from src.features.rfm_builder import RFMBuilder
        builder = RFMBuilder(rfm_config)
        rfm = builder.compute_rfm(mock_transactions, analysis_end_day=525)
        summary = builder.compute_rfm_summary(rfm)
        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
