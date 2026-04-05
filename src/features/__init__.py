"""
src.features — Feature engineering module for CLV pipeline.
Contains RFM aggregation, demographic handling, and temporal splitting.
"""

from src.features.rfm_builder import RFMBuilder
from src.features.demographic_handler import DemographicHandler
from src.features.time_splitter import TimeSplitter

__all__ = ["RFMBuilder", "DemographicHandler", "TimeSplitter"]
