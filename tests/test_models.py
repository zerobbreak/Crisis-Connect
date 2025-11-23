import pytest
from pydantic import ValidationError
from models.model import AlertModel, LocationRequest, WeatherEntry, WeatherBatch

class TestAlertModel:
    def test_valid_alert_model(self):
        """Test that a valid AlertModel can be created"""
        alert = AlertModel(
            location="Test Location",
            risk_level="High",
            message="Test alert message",
            language="en",
            timestamp="2023-10-15 12:00:00"
        )
        
        assert alert.location == "Test Location"
        assert alert.risk_level == "High"
        assert alert.message == "Test alert message"
        assert alert.language == "en"
        assert alert.timestamp == "2023-10-15 12:00:00"

class TestLocationRequest:
    def test_valid_location_request_with_place_name(self):
        """Test that a valid LocationRequest with place_name can be created"""
        location = LocationRequest(
            place_name="Durban",
            is_coastal=True
        )
        
        assert location.place_name == "Durban"
        assert location.district is None
        assert location.lat is None
        assert location.lon is None
        assert location.is_coastal is True
    
    def test_valid_location_request_with_district(self):
        """Test that a valid LocationRequest with district can be created"""
        location = LocationRequest(
            district="eThekwini (Durban)",
            is_coastal=True
        )
        
        assert location.place_name is None
        assert location.district == "eThekwini (Durban)"
        assert location.lat is None
        assert location.lon is None
        assert location.is_coastal is True
    
    def test_valid_location_request_with_coordinates(self):
        """Test that a valid LocationRequest with coordinates can be created"""
        location = LocationRequest(
            place_name="Custom Location",
            lat=-29.8587,
            lon=31.0218,
            is_coastal=False
        )
        
        assert location.place_name == "Custom Location"
        assert location.district is None
        assert location.lat == -29.8587
        assert location.lon == 31.0218
        assert location.is_coastal is False
    
    def test_invalid_location_request_missing_location(self):
        """Test that LocationRequest raises error when both place_name and district are missing"""
        with pytest.raises(ValidationError):
            LocationRequest(
                lat=-29.8587,
                lon=31.0218
            )

class TestWeatherEntry:
    def test_valid_weather_entry(self):
        """Test that a valid WeatherEntry can be created"""
        entry = WeatherEntry(
            temperature=25.5,
            humidity=65.0,
            rainfall=2.5,
            wind_speed=15.2,
            wave_height=1.2,
            location="Test Location",
            timestamp="2023-10-15 12:00:00",
            latitude=-29.8587,
            longitude=31.0218
        )
        
        assert entry.temperature == 25.5
        assert entry.humidity == 65.0
        assert entry.rainfall == 2.5
        assert entry.wind_speed == 15.2
        assert entry.wave_height == 1.2
        assert entry.location == "Test Location"
        assert entry.timestamp == "2023-10-15 12:00:00"
        assert entry.latitude == -29.8587
        assert entry.longitude == 31.0218
    
    def test_weather_entry_with_optional_fields(self):
        """Test that a WeatherEntry can be created with only required fields"""
        entry = WeatherEntry(
            location="Test Location",
            timestamp="2023-10-15 12:00:00"
        )
        
        assert entry.temperature is None
        assert entry.humidity is None
        assert entry.rainfall is None
        assert entry.wind_speed is None
        assert entry.wave_height is None
        assert entry.location == "Test Location"
        assert entry.timestamp == "2023-10-15 12:00:00"
        assert entry.latitude is None
        assert entry.longitude is None

class TestWeatherBatch:
    def test_valid_weather_batch(self):
        """Test that a valid WeatherBatch can be created"""
        entries = [
            WeatherEntry(
                temperature=25.5,
                humidity=65.0,
                rainfall=2.5,
                wind_speed=15.2,
                wave_height=1.2,
                location="Test Location 1",
                timestamp="2023-10-15 12:00:00"
            ),
            WeatherEntry(
                temperature=22.3,
                humidity=70.5,
                rainfall=0.0,
                wind_speed=10.8,
                wave_height=0.5,
                location="Test Location 2",
                timestamp="2023-10-15 12:00:00"
            )
        ]
        
        batch = WeatherBatch(data=entries)
        
        assert len(batch.data) == 2
        assert batch.data[0].location == "Test Location 1"
        assert batch.data[1].location == "Test Location 2"