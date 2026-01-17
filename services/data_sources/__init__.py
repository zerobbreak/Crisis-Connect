"""
Data source fetchers for Crisis Connect Phase 1

This module provides data fetchers for collecting disaster and weather data
from multiple external sources.
"""

from .emdat_fetcher import EMDATFetcher
from .noaa_fetcher import NOAAFetcher

__all__ = ["EMDATFetcher", "NOAAFetcher"]
