"""
test_clv_models.py — Tests for CLVModeler
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def model_config(tmp_path):
    """Minimal config for CLVModeler."""
    return {
        "project": {"random_seed": 42},
        "data": {"processed_dir": str(tmp_path / "processed")},
        "output": {"model_dir": str(tmp_path / "models")},
        "model": {
            "active": "bgnbd_gg",
            "bgnbd_gg": {"penalizer_coef": 0.001},
            "xgboost": {
                "n_estimators": 50, "max_depth": 3,
                "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "lightgbm": {
                "n_estimators": 50, "max_depth": 3,
                "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
    }


@pytest.fixture
def mock_rfm():
    """Create mock RFM data for model testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "household_key": range(1, n + 1),
        "Recency": np.random.randint(0, 50, n),
        "Frequency": np.random.randint(0, 30, n),
        "T": np.random.randint(10, 75, n),
        "Gross_Sales": np.random.uniform(10, 500, n).round(2),
        "Net_Sales": np.random.uniform(8, 400, n).round(2),
        "avg_monetary": np.random.uniform(5, 50, n).round(2),
        "total_baskets": np.random.randint(1, 31, n),
        "avg_basket_size": np.random.uniform(1, 10, n).round(2),
        "avg_transaction_value": np.random.uniform(5, 50, n).round(2),
        "distinct_stores": np.random.randint(1, 5, n),
        "tenure_days": np.random.randint(0, 490, n),
        "coupon_usage_rate": np.random.uniform(0, 0.5, n).round(4),
        "first_purchase_week": np.random.randint(1, 30, n),
        "last_purchase_week": np.random.randint(30, 75, n),
    })


class TestCLVModeler:
    """Tests for CLVModeler."""

    def test_segment_customers_adds_cluster_column(self, model_config, mock_rfm):
        """Test that segmentation adds Cluster and Segment columns."""
        from src.models.clv_models import CLVModeler
        modeler = CLVModeler(model_config)
        result = modeler.segment_customers(mock_rfm)
        assert "Cluster" in result.columns
        assert "Segment" in result.columns

    def test_segment_labels_are_strings(self, model_config, mock_rfm):
        """Test segment labels are human-readable strings."""
        from src.models.clv_models import CLVModeler
        modeler = CLVModeler(model_config)
        result = modeler.segment_customers(mock_rfm)
        assert result["Segment"].dtype == object
        # All values should be non-empty strings
        assert result["Segment"].str.len().min() > 0

    def test_kmeans_uses_consistent_features(self, model_config, mock_rfm):
        """Test K-Means uses avg_monetary (not Net_Sales)."""
        from src.models.clv_models import CLVModeler
        modeler = CLVModeler(model_config)
        modeler.segment_customers(mock_rfm)
        # Verify scaler was fitted on 3 features
        assert modeler.scaler is not None
        assert modeler.scaler.n_features_in_ == 3

    def test_fit_bgnbd_adds_predictions(self, model_config, mock_rfm):
        """Test BG/NBD fitting adds predicted purchases column."""
        pytest.importorskip("lifetimes")
        from src.models.clv_models import CLVModeler
        modeler = CLVModeler(model_config)
        result = modeler.fit_bgnbd(mock_rfm)
        assert "predicted_purchases_6m" in result.columns
        assert (result["predicted_purchases_6m"] >= 0).all()

    def test_save_models_creates_files(self, model_config, mock_rfm):
        """Test that save_models creates pkl files."""
        pytest.importorskip("lifetimes")
        from src.models.clv_models import CLVModeler
        from pathlib import Path
        modeler = CLVModeler(model_config)
        modeler.segment_customers(mock_rfm)
        modeler.fit_bgnbd(mock_rfm)
        modeler.save_models()

        model_dir = Path(model_config["output"]["model_dir"])
        assert (model_dir / "kmeans_model.pkl").exists()
        assert (model_dir / "bgnbd_model.pkl").exists()

    def test_customer_labels_csv_created(self, model_config, mock_rfm):
        """Test customer_labels.csv is saved with Segment column."""
        from src.models.clv_models import CLVModeler
        from pathlib import Path
        modeler = CLVModeler(model_config)
        modeler.segment_customers(mock_rfm)

        labels_path = Path(model_config["data"]["processed_dir"]) / "customer_labels.csv"
        assert labels_path.exists()

        labels_df = pd.read_csv(labels_path)
        assert "Segment" in labels_df.columns
        assert "household_key" in labels_df.columns
