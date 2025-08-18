from pydantic import BaseModel, model_validator
from typing import List, Optional

class AlertModel(BaseModel):
    location: str
    risk_level: str
    message: str
    language: str
    timestamp: str

from typing import Optional
from pydantic import BaseModel, field_validator, model_validator

class LocationRequest(BaseModel):
    place_name: Optional[str] = None
    district: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    is_coastal: Optional[bool] = False

    # ✅ Use model_validator for cross-field checks
    @model_validator(mode="after")
    def check_at_least_one_location(self):
        if not self.place_name and not self.district:
            raise ValueError("At least one of place_name or district must be provided")
        return self


class WeatherEntry(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: Optional[float] = None
    wind_speed: Optional[float] = None
    wave_height: Optional[float] = None
    location: str
    timestamp: str  # Changed to str to match Prisma schema
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class WeatherBatch(BaseModel):
    data: List[WeatherEntry]