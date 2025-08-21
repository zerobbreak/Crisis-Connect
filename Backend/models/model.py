from pydantic import BaseModel, model_validator, field_validator, constr, confloat,Field, ConfigDict
from typing import List, Optional, Literal
from datetime import datetime

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
