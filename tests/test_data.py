"""
test_data.py
============
Unit tests for data ingestion module (src.data.make_dataset).
"""

import pandas as pd
import pytest

from src.data.make_dataset import load_config, validate_schema


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_valid_config(self, tmp_path):
        """Ensure a valid YAML config is parsed correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "project:\n  name: test\ndata:\n  raw_file: data/raw/test.csv\n"
        )
        config = load_config(str(config_file))
        assert config["project"]["name"] == "test"

    def test_load_missing_config(self):
        """Ensure FileNotFoundError on missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestValidateSchema:
    """Tests for schema validation."""

    def test_valid_schema(self):
        """Passes when all required columns are present."""
        df = pd.DataFrame({"datetime": [1], "concurrent_players": [100]})
        config = {"data": {"columns": {"datetime": "datetime", "target": "concurrent_players"}}}
        validate_schema(df, config)  # Should not raise

    def test_missing_column(self):
        """Raises ValueError when a required column is missing."""
        df = pd.DataFrame({"datetime": [1]})
        config = {"data": {"columns": {"datetime": "datetime", "target": "concurrent_players"}}}
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df, config)
