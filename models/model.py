from pydantic import BaseModel, model_validator, field_validator, constr, confloat,Field, ConfigDict
from typing import List, Optional, Literal, Dict, An
from datetime import datetime
import uuid

class AlertModel(BaseModel):
    location: str = Field(
        ..., min_length=1, max_length=100, 
        description="Location name (e.g., eThekwini)"
    )
    risk_level: Literal["LOW", "MODERATE", "HIGH", "SEVERE"] = Field(
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

    @field_validator('risk_level')
    @classmethod
    def normalize_risk_level(cls, v):
        return v.upper()

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

class WeatherEntry(BaseModel):
    temperature: Optional[confloat(ge=-100.0, le=100.0)] = None
    humidity: Optional[confloat(ge=0.0, le=100.0)] = None
    rainfall: Optional[confloat(ge=0.0, le=5000.0)] = None
    wind_speed: Optional[confloat(ge=0.0, le=500.0)] = None
    wave_height: Optional[confloat(ge=0.0, le=50.0)] = None
    location: constr(min_length=1, max_length=100, strip_whitespace=True)
    timestamp: str
    latitude: Optional[confloat(ge=-90.0, le=90.0)] = None
    longitude: Optional[confloat(ge=-180.0, le=180.0)] = None
    
    @field_validator('timestamp')
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

class LocationCreate(BaseModel):
    """Model for creating new locations"""
    name: constr(min_length=1, max_length=100, strip_whitespace=True) = Field(
        ..., description="Location name"
    )
    display_name: Optional[constr(min_length=1, max_length=200, strip_whitespace=True)] = Field(
        None, description="Human-readable display name"
    )
    latitude: confloat(ge=-90.0, le=90.0) = Field(
        ..., description="Latitude coordinate"
    )
    longitude: confloat(ge=-180.0, le=180.0) = Field(
        ..., description="Longitude coordinate"
    )
    country: Optional[constr(min_length=2, max_length=50, strip_whitespace=True)] = Field(
        "South Africa", description="Country name"
    )
    region: Optional[constr(min_length=1, max_length=100, strip_whitespace=True)] = Field(
        None, description="Region or province"
    )
    district: Optional[constr(min_length=1, max_length=100, strip_whitespace=True)] = Field(
        None, description="District or municipality"
    )
    is_coastal: bool = Field(
        False, description="Whether location is coastal"
    )
    elevation_meters: Optional[confloat(ge=-500, le=9000)] = Field(
        None, description="Elevation in meters"
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
    """Model for updating locations"""
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

class Location(BaseModel):
    """Full location model with database fields"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    display_name: Optional[str] = None
    latitude: float
    longitude: float
    country: str = "South Africa"
    region: Optional[str] = None
    district: Optional[str] = None
    is_coastal: bool = False
    elevation_meters: Optional[float] = None
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
