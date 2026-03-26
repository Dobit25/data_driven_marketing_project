"""
test_models.py
==============
Unit tests for evaluation utilities (src.models.predict_model).
"""

import numpy as np
import pytest

from src.models.predict_model import compute_metrics


class TestComputeMetrics:
    """Tests for metric computation."""

    def test_perfect_prediction(self):
        """All metrics should be zero for perfect predictions."""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([100, 200, 300, 400, 500])
        metrics = compute_metrics(actual, predicted)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["mape"] == pytest.approx(0.0)

    def test_known_error(self):
        """Verify metric values for a known error case."""
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 190.0])
        metrics = compute_metrics(actual, predicted)
        assert metrics["rmse"] == pytest.approx(10.0)
        assert metrics["mae"] == pytest.approx(10.0)
        assert metrics["mape"] == pytest.approx(7.5)  # (10/100 + 10/200)/2 * 100

    def test_metrics_are_positive(self):
        """RMSE and MAE should always be non-negative."""
        rng = np.random.default_rng(42)
        actual = rng.uniform(100, 1000, size=50)
        predicted = actual + rng.normal(0, 50, size=50)
        metrics = compute_metrics(actual, predicted)
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["mape"] >= 0
