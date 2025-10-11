"""
Enhanced Historical Data Models for Crisis Connect
Comprehensive flood-specific data structures for better historical tracking
"""

from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime, date
from enum import Enum
import re


class FloodSeverityLevel(str, Enum):
    """Flood severity classification"""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"
    CATASTROPHIC = "catastrophic"


class FloodType(str, Enum):
    """Types of flood events"""
    FLASH_FLOOD = "flash_flood"
    RIVER_FLOOD = "river_flood"
    COASTAL_FLOOD = "coastal_flood"
    URBAN_FLOOD = "urban_flood"
    DAM_BREAK = "dam_break"
    STORM_SURGE = "storm_surge"
    SEASONAL_FLOOD = "seasonal_flood"


class ImpactCategory(str, Enum):
    """Impact categories for flood events"""
    HUMAN = "human"
    INFRASTRUCTURE = "infrastructure"
    ECONOMIC = "economic"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"


class WeatherConditions(BaseModel):
    """Detailed weather conditions during flood event"""
    temperature_c: Optional[float] = Field(None, ge=-50, le=50)
    humidity_percent: Optional[float] = Field(None, ge=0, le=100)
    rainfall_mm: Optional[float] = Field(None, ge=0, le=1000)
    wind_speed_kmh: Optional[float] = Field(None, ge=0, le=300)
    wind_direction_degrees: Optional[float] = Field(None, ge=0, le=360)
    pressure_mb: Optional[float] = Field(None, ge=950, le=1050)
    visibility_km: Optional[float] = Field(None, ge=0, le=50)
    cloud_cover_percent: Optional[float] = Field(None, ge=0, le=100)
    wave_height_m: Optional[float] = Field(None, ge=0, le=20)
    sea_level_pressure_mb: Optional[float] = Field(None, ge=950, le=1050)
    
    # Derived weather features
    precipitation_intensity: Optional[Literal["light", "moderate", "heavy", "extreme"]] = None
    storm_duration_hours: Optional[float] = Field(None, ge=0, le=168)  # Max 1 week
    antecedent_rainfall_7days: Optional[float] = Field(None, ge=0, le=1000)
    antecedent_rainfall_30days: Optional[float] = Field(None, ge=0, le=2000)


class GeographicLocation(BaseModel):
    """Enhanced geographic information"""
    name: str = Field(..., min_length=1, max_length=200)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation_m: Optional[float] = Field(None, ge=-500, le=5000)
    district: Optional[str] = Field(None, max_length=100)
    province: Optional[str] = Field(None, max_length=100)
    country: str = Field(default="South Africa", max_length=100)
    
    # Hydrological features
    nearest_river: Optional[str] = Field(None, max_length=100)
    river_distance_km: Optional[float] = Field(None, ge=0, le=100)
    watershed_area_km2: Optional[float] = Field(None, ge=0, le=10000)
    drainage_density: Optional[float] = Field(None, ge=0, le=10)
    
    # Urban characteristics
    population_density: Optional[float] = Field(None, ge=0, le=50000)
    urbanization_level: Optional[Literal["rural", "suburban", "urban", "metropolitan"]] = None
    land_use_type: Optional[Literal["residential", "commercial", "industrial", "agricultural", "natural"]] = None


class ImpactMetrics(BaseModel):
    """Quantified impact measurements"""
    # Human impact
    deaths: int = Field(default=0, ge=0, le=10000)
    injuries: int = Field(default=0, ge=0, le=50000)
    displaced_persons: int = Field(default=0, ge=0, le=100000)
    evacuated_persons: int = Field(default=0, ge=0, le=100000)
    
    # Infrastructure impact
    damaged_homes: int = Field(default=0, ge=0, le=100000)
    destroyed_homes: int = Field(default=0, ge=0, le=100000)
    damaged_roads_km: Optional[float] = Field(None, ge=0, le=10000)
    damaged_bridges: int = Field(default=0, ge=0, le=1000)
    damaged_utilities: int = Field(default=0, ge=0, le=10000)
    
    # Economic impact (in USD)
    property_damage_usd: Optional[float] = Field(None, ge=0, le=10000000000)
    infrastructure_damage_usd: Optional[float] = Field(None, ge=0, le=10000000000)
    business_loss_usd: Optional[float] = Field(None, ge=0, le=10000000000)
    agricultural_loss_usd: Optional[float] = Field(None, ge=0, le=1000000000)
    total_economic_impact_usd: Optional[float] = Field(None, ge=0, le=20000000000)
    
    # Environmental impact
    contaminated_water_sources: Optional[int] = Field(None, ge=0, le=1000)
    soil_erosion_hectares: Optional[float] = Field(None, ge=0, le=100000)
    wildlife_impact: Optional[Literal["minimal", "moderate", "significant", "severe"]] = None


class ResponseMetrics(BaseModel):
    """Emergency response and recovery metrics"""
    response_time_hours: Optional[float] = Field(None, ge=0, le=168)
    evacuation_time_hours: Optional[float] = Field(None, ge=0, le=168)
    rescue_operations_count: Optional[int] = Field(None, ge=0, le=10000)
    emergency_shelters_opened: Optional[int] = Field(None, ge=0, le=1000)
    aid_distributed_usd: Optional[float] = Field(None, ge=0, le=1000000000)
    recovery_time_days: Optional[int] = Field(None, ge=0, le=3650)  # Max 10 years
    
    # Effectiveness metrics
    early_warning_effectiveness: Optional[Literal["none", "poor", "fair", "good", "excellent"]] = None
    community_preparedness: Optional[Literal["none", "poor", "fair", "good", "excellent"]] = None
    infrastructure_resilience: Optional[Literal["none", "poor", "fair", "good", "excellent"]] = None


class PredictiveFeatures(BaseModel):
    """Features derived for ML model improvement"""
    # Weather anomaly scores
    temperature_anomaly_score: Optional[float] = Field(None, ge=-5, le=5)
    rainfall_anomaly_score: Optional[float] = Field(None, ge=-5, le=5)
    humidity_anomaly_score: Optional[float] = Field(None, ge=-5, le=5)
    wind_anomaly_score: Optional[float] = Field(None, ge=-5, le=5)
    
    # Hydrological indicators
    soil_moisture_index: Optional[float] = Field(None, ge=0, le=1)
    river_level_percentile: Optional[float] = Field(None, ge=0, le=100)
    groundwater_level_change: Optional[float] = Field(None, ge=-10, le=10)
    
    # Seasonal factors
    seasonal_risk_factor: Optional[float] = Field(None, ge=0, le=1)
    monsoon_intensity: Optional[float] = Field(None, ge=0, le=1)
    storm_frequency_factor: Optional[float] = Field(None, ge=0, le=1)
    
    # Model predictions (for validation)
    predicted_risk_score: Optional[float] = Field(None, ge=0, le=100)
    predicted_severity: Optional[FloodSeverityLevel] = None
    prediction_accuracy: Optional[float] = Field(None, ge=0, le=1)


class HistoricalFloodEvent(BaseModel):
    """Comprehensive historical flood event record"""
    
    # Event identification
    event_id: str = Field(..., min_length=1, max_length=50)
    name: Optional[str] = Field(None, max_length=200)
    alternative_names: Optional[List[str]] = Field(None, max_items=10)
    
    # Temporal information
    start_date: date
    end_date: Optional[date] = None
    duration_hours: Optional[float] = Field(None, ge=0, le=8760)  # Max 1 year
    peak_intensity_time: Optional[datetime] = None
    
    # Classification
    flood_type: FloodType
    severity: FloodSeverityLevel
    primary_cause: Optional[str] = Field(None, max_length=200)
    contributing_factors: Optional[List[str]] = Field(None, max_items=10)
    
    # Geographic information
    location: GeographicLocation
    affected_area_km2: Optional[float] = Field(None, ge=0, le=1000000)
    flood_depth_m: Optional[float] = Field(None, ge=0, le=50)
    flood_velocity_ms: Optional[float] = Field(None, ge=0, le=50)
    
    # Weather conditions
    weather_conditions: Optional[WeatherConditions] = None
    
    # Impact assessment
    impacts: ImpactMetrics
    
    # Response and recovery
    response: Optional[ResponseMetrics] = None
    
    # Predictive features
    predictive_features: Optional[PredictiveFeatures] = None
    
    # Metadata
    data_source: str = Field(..., max_length=200)
    data_quality: Literal["excellent", "good", "fair", "poor"] = Field(default="good")
    last_updated: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = Field(None, max_length=100)
    
    # Validation and verification
    verified: bool = Field(default=False)
    verification_source: Optional[str] = Field(None, max_length=200)
    verification_date: Optional[datetime] = None
    
    # Additional context
    description: Optional[str] = Field(None, max_length=2000)
    lessons_learned: Optional[str] = Field(None, max_length=2000)
    recommendations: Optional[List[str]] = Field(None, max_items=20)
    
    @validator('event_id')
    def validate_event_id(cls, v):
        if not re.match(r'^[A-Z0-9_-]+$', v):
            raise ValueError('Event ID must contain only uppercase letters, numbers, hyphens, and underscores')
        return v
    
    @validator('end_date')
    def validate_end_date(cls, v, values):
        if v and 'start_date' in values and v < values['start_date']:
            raise ValueError('End date must be after start date')
        return v
    
    @model_validator(mode='after')
    def validate_duration(self):
        start_date = self.start_date
        end_date = self.end_date
        duration = self.duration_hours
        
        if start_date and end_date and duration:
            calculated_duration = (end_date - start_date).total_seconds() / 3600
            if abs(duration - calculated_duration) > 24:  # Allow 24-hour tolerance
                raise ValueError('Duration hours must be consistent with start and end dates')
        
        return self


class HistoricalSummary(BaseModel):
    """Summary statistics for historical data analysis"""
    location: GeographicLocation
    total_events: int = Field(ge=0)
    events_by_type: Dict[FloodType, int] = Field(default_factory=dict)
    events_by_severity: Dict[FloodSeverityLevel, int] = Field(default_factory=dict)
    
    # Temporal patterns
    events_by_month: Dict[int, int] = Field(default_factory=dict)  # 1-12
    events_by_year: Dict[int, int] = Field(default_factory=dict)
    
    # Impact statistics
    total_deaths: int = Field(default=0)
    total_injuries: int = Field(default=0)
    total_displaced: int = Field(default=0)
    total_property_damage_usd: float = Field(default=0)
    
    # Risk indicators
    average_severity_score: Optional[float] = Field(None, ge=0, le=5)
    flood_frequency_per_year: Optional[float] = Field(None, ge=0, le=365)
    risk_trend: Optional[Literal["decreasing", "stable", "increasing"]] = None
    
    # Data quality
    data_completeness_percent: float = Field(default=0, ge=0, le=100)
    last_event_date: Optional[date] = None
    first_event_date: Optional[date] = None
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    calculation_method: str = Field(default="automatic")


class FloodEventSearch(BaseModel):
    """Search criteria for historical flood events"""
    location_name: Optional[str] = None
    district: Optional[str] = None
    province: Optional[str] = None
    
    # Date range
    start_date_from: Optional[date] = None
    start_date_to: Optional[date] = None
    
    # Severity and type filters
    severity_levels: Optional[List[FloodSeverityLevel]] = None
    flood_types: Optional[List[FloodType]] = None
    
    # Impact thresholds
    min_deaths: Optional[int] = Field(None, ge=0)
    min_damage_usd: Optional[float] = Field(None, ge=0)
    
    # Data quality
    min_data_quality: Optional[Literal["excellent", "good", "fair", "poor"]] = None
    verified_only: bool = Field(default=False)
    
    # Pagination
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    
    # Sorting
    sort_by: Optional[Literal["start_date", "severity", "deaths", "damage_usd"]] = Field(default="start_date")
    sort_order: Literal["asc", "desc"] = Field(default="desc")


class FloodEventUpdate(BaseModel):
    """Model for updating historical flood events"""
    name: Optional[str] = None
    severity: Optional[FloodSeverityLevel] = None
    flood_type: Optional[FloodType] = None
    description: Optional[str] = None
    impacts: Optional[ImpactMetrics] = None
    response: Optional[ResponseMetrics] = None
    predictive_features: Optional[PredictiveFeatures] = None
    data_quality: Optional[Literal["excellent", "good", "fair", "poor"]] = None
    verified: Optional[bool] = None
    verification_source: Optional[str] = None
    lessons_learned: Optional[str] = None
    recommendations: Optional[List[str]] = None
