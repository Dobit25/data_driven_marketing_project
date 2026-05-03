"""
src.features — Feature engineering module for CLV pipeline.
Contains RFM aggregation, demographic handling, temporal splitting,
market basket analysis, and promotional feature extraction.
"""

from src.features.rfm_builder import RFMBuilder
from src.features.demographic_handler import DemographicHandler
from src.features.time_splitter import TimeSplitter
from src.features.mba_builder import MBABuilder
from src.features.causal_features import CausalFeatureBuilder

__all__ = [
    "RFMBuilder", "DemographicHandler", "TimeSplitter",
    "MBABuilder", "CausalFeatureBuilder",
]

