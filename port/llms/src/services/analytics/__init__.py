"""
Analytics service module with focused classes for data fetching, processing, and visualization.
"""
from .data_fetcher import AnalyticsDataFetcher
from .data_processor import AnalyticsDataProcessor
from .color_manager import AnalyticsColorManager
from .analytics_service import AnalyticsService

__all__ = [
    'AnalyticsDataFetcher',
    'AnalyticsDataProcessor',
    'AnalyticsColorManager',
    'AnalyticsService'
]

