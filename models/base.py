"""
Shared base models and enums for Crisis Connect
Used by both operational and historical models
"""

from pydantic import BaseModel, Field, constr, confloat
from typing import Optional, Literal
from enum import Enum


# ============= Enums =============

class RiskLevel(str, Enum):
    """Unified risk level classification"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    CRITICAL = "CRITICAL"  # For extreme cases


class FloodType(str, Enum):
    """Types of flood events"""
    FLASH_FLOOD = "flash_flood"
    RIVER_FLOOD = "river_flood"
    COASTAL_FLOOD = "coastal_flood"
    URBAN_FLOOD = "urban_flood"
    DAM_BREAK = "dam_break"
    STORM_SURGE = "storm_surge"
    SEASONAL_FLOOD = "seasonal_flood"


class SeverityLevel(str, Enum):
    """Flood severity classification (for historical data)"""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"
    CATASTROPHIC = "catastrophic"


# ============= Base Location Model =============

class BaseLocation(BaseModel):
    """Shared location fields"""
    name: constr(min_length=1, max_length=100, strip_whitespace=True)
    latitude: confloat(ge=-90.0, le=90.0)
    longitude: confloat(ge=-180.0, le=180.0)
    country: str = Field(default="South Africa", max_length=100)
    region: Optional[str] = Field(None, max_length=100)
    district: Optional[str] = Field(None, max_length=100)
    is_coastal: bool = False
    elevation_meters: Optional[float] = Field(None, ge=-500, le=9000)


# ============= Base Weather Model =============

class BaseWeather(BaseModel):
    """Shared weather fields"""
    temperature_c: Optional[float] = Field(None, ge=-100.0, le=100.0)
    humidity_percent: Optional[float] = Field(None, ge=0.0, le=100.0)
    rainfall_mm: Optional[float] = Field(None, ge=0.0, le=5000.0)
    wind_speed_kmh: Optional[float] = Field(None, ge=0.0, le=500.0)
    pressure_mb: Optional[float] = Field(None, ge=950, le=1050)


# ============= Validation Helpers =============

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate coordinate ranges"""
    return -90 <= lat <= 90 and -180 <= lon <= 180
