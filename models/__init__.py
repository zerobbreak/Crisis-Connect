"""
Crisis Connect Data Models
Centralized export of all Pydantic models
"""

# Base models and enums
from .base import (
    RiskLevel,
    FloodType,
    SeverityLevel,
    BaseLocation,
    BaseWeather,
    validate_coordinates
)

# Operational models (real-time)
from .model import (
    AlertModel,
    LocationRequest,
    WeatherEntry,
    WeatherBatch,
    SimulateRequest,
    LocationCreate,
    LocationUpdate,
    Location,
    LocationPreset,
    LocationSearch,
    GeocodeRequest
)

# Historical models (archival/analysis)
from .historical_models import (
    FloodSeverityLevel,
    FloodType as HistoricalFloodType,  # Alias to avoid conflict
    ImpactCategory,
    WeatherConditions,
    GeographicLocation,
    ImpactMetrics,
    ResponseMetrics,
    PredictiveFeatures,
    HistoricalFloodEvent,
    HistoricalSummary,
    FloodEventSearch,
    FloodEventUpdate
)

__all__ = [
    # Base
    "RiskLevel",
    "FloodType",
    "SeverityLevel",
    "BaseLocation",
    "BaseWeather",
    "validate_coordinates",
    
    # Operational
    "AlertModel",
    "LocationRequest",
    "WeatherEntry",
    "WeatherBatch",
    "SimulateRequest",
    "LocationCreate",
    "LocationUpdate",
    "Location",
    "LocationPreset",
    "LocationSearch",
    "GeocodeRequest",
    
    # Historical
    "FloodSeverityLevel",
    "HistoricalFloodType",
    "ImpactCategory",
    "WeatherConditions",
    "GeographicLocation",
    "ImpactMetrics",
    "ResponseMetrics",
    "PredictiveFeatures",
    "HistoricalFloodEvent",
    "HistoricalSummary",
    "FloodEventSearch",
    "FloodEventUpdate",
]
