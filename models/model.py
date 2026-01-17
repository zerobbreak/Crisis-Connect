from pydantic import BaseModel, model_validator, field_validator, constr, confloat, Field, ConfigDict
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
import uuid

# Import shared base models
from .base import RiskLevel, BaseLocation, BaseWeather

class AlertModel(BaseModel):
    location: str = Field(
        ..., min_length=1, max_length=100, 
        description="Location name (e.g., eThekwini)"
    )
    risk_level: RiskLevel = Field(
        ..., description="Severity level"
    )
    message: str = Field(
        ..., min_length=1, max_length=1000,
        description="Alert message for public"
    )
    language: str = Field(
        ..., pattern=r'^[a-z]{2,10}$', 
        description="Language code (e.g., 'en', 'zu')"
    )
    timestamp: datetime = Field(
        ..., description="ISO 8601 timestamp"
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "location": "eThekwini",
                "risk_level": "HIGH",
                "message": "Flood warning: Evacuate low-lying areas.",
                "language": "en",
                "timestamp": "2025-04-05 10:30:00"
            }
        }
    )

class LocationRequest(BaseModel):
    place_name: Optional[constr(min_length=1, max_length=100, strip_whitespace=True)] = None
    district: Optional[constr(min_length=1, max_length=100, strip_whitespace=True)] = None
    lat: Optional[confloat(ge=-90.0, le=90.0)] = None
    lon: Optional[confloat(ge=-180.0, le=180.0)] = None
    is_coastal: Optional[bool] = False

    @model_validator(mode="after")
    def check_at_least_one_location(self):
        if not self.place_name and not self.district and (self.lat is None or self.lon is None):
            raise ValueError("At least one location identifier must be provided")
        return self

class WeatherEntry(BaseWeather):
    """Real-time weather entry extending base weather fields"""
    # Additional fields not in BaseWeather
    wave_height: Optional[confloat(ge=0.0, le=50.0)] = None
    location: constr(min_length=1, max_length=100, strip_whitespace=True)
    timestamp: str
    latitude: Optional[confloat(ge=-90.0, le=90.0)] = None
    longitude: Optional[confloat(ge=-180.0, le=180.0)] = None
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            return v
        except ValueError:
            raise ValueError("Timestamp must be in format YYYY-MM-DD HH:MM:SS")

class WeatherBatch(BaseModel):
    data: List[WeatherEntry]

class SimulateRequest(BaseModel):
    location: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    scenario: Literal["flood", "storm", "coastal_cyclone", "flash_flood"] = "flood"
    duration_hours: int = 24
    household_size: int = 4

# --- Dynamic Location Management Models ---

class LocationCreate(BaseLocation):
    """Model for creating new locations - extends BaseLocation with operational fields"""
    display_name: Optional[constr(min_length=1, max_length=200, strip_whitespace=True)] = Field(
        None, description="Human-readable display name"
    )
    population: Optional[int] = Field(
        None, ge=0, description="Approximate population"
    )
    tags: List[str] = Field(
        default_factory=list, description="Custom tags for categorization"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="allow",
        json_schema_extra={
            "example": {
                "name": "Cape Town",
                "display_name": "Cape Town Metropolitan",
                "latitude": -33.9249,
                "longitude": 18.4241,
                "country": "South Africa",
                "region": "Western Cape",
                "district": "Cape Town",
                "is_coastal": True,
                "elevation_meters": 10,
                "population": 433688,
                "tags": ["metropolitan", "coastal", "tourist"],
                "metadata": {"timezone": "Africa/Johannesburg"}
            }
        }
    )


class LocationUpdate(BaseModel):
    """Model for updating locations - all fields optional"""
    name: Optional[constr(min_length=1, max_length=100, strip_whitespace=True)] = None
    display_name: Optional[constr(min_length=1, max_length=200, strip_whitespace=True)] = None
    latitude: Optional[confloat(ge=-90.0, le=90.0)] = None
    longitude: Optional[confloat(ge=-180.0, le=180.0)] = None
    country: Optional[constr(min_length=2, max_length=50, strip_whitespace=True)] = None
    region: Optional[constr(min_length=1, max_length=100, strip_whitespace=True)] = None
    district: Optional[constr(min_length=1, max_length=100, strip_whitespace=True)] = None
    is_coastal: Optional[bool] = None
    elevation_meters: Optional[confloat(ge=-500, le=9000)] = None
    population: Optional[int] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = True

class Location(BaseLocation):
    """Full location model with database fields - extends BaseLocation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    display_name: Optional[str] = None
    population: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_weather_update: Optional[datetime] = None

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="allow"
    )

class LocationPreset(BaseModel):
    """Model for location presets/templates"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: constr(min_length=1, max_length=100, strip_whitespace=True)
    description: Optional[str] = None
    locations: List[str] = Field(..., description="List of location IDs")
    category: Literal["country", "region", "province", "district", "custom", "emergency"] = "custom"
    is_public: bool = True
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class LocationSearch(BaseModel):
    """Model for location search/filtering"""
    name: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    district: Optional[str] = None
    is_coastal: Optional[bool] = None
    tags: Optional[List[str]] = None
    latitude_min: Optional[float] = None
    latitude_max: Optional[float] = None
    longitude_min: Optional[float] = None
    longitude_max: Optional[float] = None
    is_active: Optional[bool] = True
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)

class GeocodeRequest(BaseModel):
    """Model for geocoding requests"""
    query: constr(min_length=1, max_length=200, strip_whitespace=True) = Field(
        ..., description="Location name or address to geocode"
    )
    country: Optional[str] = "South Africa"
    limit: int = Field(5, ge=1, le=20)
