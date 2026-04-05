"""
src.data — Data loading and ingestion module for Dunnhumby dataset.
Handles optimized CSV reading with dtype downcasting and schema validation.
"""

from src.data.data_loader import DunnhumbyDataLoader

__all__ = ["DunnhumbyDataLoader"]
