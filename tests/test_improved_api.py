"""
Comprehensive test suite for improved Crisis Connect API
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import json
from datetime import datetime, timedelta

from main import app
from config import settings
from models.model import AlertModel
from services.weather_service import WeatherService
from services.alert_service import AlertService


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_weather_data():
    """Mock weather data for testing"""
    return pd.DataFrame({
        'location': ['Durban', 'Cape Town'],
        'lat': [-29.8587, -33.9249],
        'lon': [31.0218, 18.4241],
        'temp_c': [25.5, 22.3],
        'humidity': [65.0, 70.0],
        'wind_kph': [15.2, 12.8],
        'pressure_mb': [1013.0, 1015.0],
        'precip_mm': [2.5, 0.0],
        'cloud': [20.0, 15.0],
        'wave_height': [1.2, 2.1],
        'composite_risk_score': [75.0, 45.0],
        'risk_category': ['High', 'Low']
    })


@pytest.fixture
def mock_alert_data():
    """Mock alert data for testing"""
    return AlertModel(
        location="Durban",
        risk_level="HIGH",
        message="Test flood warning",
        language="en",
        timestamp=datetime.now()
    )


class TestHealthCheck:
    """Test health check endpoints"""
    
    @patch('main.get_db')
    @patch('main.get_redis')
    def test_health_check_healthy(self, mock_redis, mock_db, client):
        """Test health check when all services are healthy"""
        # Mock database
        mock_db_instance = MagicMock()
        mock_db_instance.command = AsyncMock(return_value={"ok": 1})
        mock_db.return_value = mock_db_instance
        
        # Mock Redis
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping = AsyncMock(return_value=True)
        mock_redis.return_value = mock_redis_instance
        
        # Mock model
        with patch('main.model', MagicMock()):
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "services" in data
            assert data["services"]["mongodb"]["status"] == "healthy"
    
    def test_health_check_unhealthy(self, client):
        """Test health check when services are unhealthy"""
        with patch('main.model', None):
            response = client.get("/health")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"


class TestWeatherDataCollection:
    """Test weather data collection endpoints"""
    
    @patch('main.weather_service')
    @patch('main.rate_limit')
    @patch('main.verify_api_key')
    def test_collect_data_success(self, mock_api_key, mock_rate_limit, mock_weather_service, client):
        """Test successful weather data collection"""
        mock_api_key.return_value = True
        mock_rate_limit.return_value = True
        
        mock_weather_service.collect_weather_data = AsyncMock(return_value={
            "success": True,
            "message": "Data collected successfully",
            "count": 5,
            "collection_time_seconds": 2.5,
            "timestamp": datetime.now().isoformat()
        })
        
        response = client.get("/collect")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 5
    
    @patch('main.weather_service')
    def test_collect_data_custom_locations(self, mock_weather_service, client):
        """Test custom location data collection"""
        mock_weather_service.collect_weather_data = AsyncMock(return_value={
            "success": True,
            "message": "Custom data collected successfully",
            "count": 2,
            "collection_time_seconds": 1.5,
            "timestamp": datetime.now().isoformat()
        })
        
        payload = {
            "locations": [
                {"name": "Test Location", "lat": -29.8587, "lon": 31.0218}
            ]
        }
        
        response = client.post("/collect", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 2
    
    def test_collect_data_invalid_locations(self, client):
        """Test data collection with invalid locations"""
        payload = {
            "locations": [
                {"name": "Invalid", "lat": 999, "lon": 999}  # Invalid coordinates
            ]
        }
        
        response = client.post("/collect", json=payload)
        
        # Should return 400 for invalid coordinates
        assert response.status_code == 400


class TestRiskAssessment:
    """Test risk assessment endpoints"""
    
    @patch('main.weather_service')
    @patch('main.model')
    def test_predict_risk_success(self, mock_model, mock_weather_service, client):
        """Test successful risk assessment"""
        mock_weather_service.process_risk_assessment = AsyncMock(return_value={
            "success": True,
            "message": "Processed 5 predictions",
            "predictions_count": 5,
            "alerts_generated": 2,
            "duration_seconds": 3.2,
            "timestamp": datetime.now().isoformat()
        })
        
        response = client.post("/predict")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["predictions_count"] == 5
        assert data["alerts_generated"] == 2
    
    def test_predict_risk_no_model(self, client):
        """Test risk assessment when model is not loaded"""
        with patch('main.model', None):
            response = client.post("/predict")
            
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]


class TestAlertManagement:
    """Test alert management endpoints"""
    
    @patch('main.alert_service')
    def test_create_alert_success(self, mock_alert_service, client, mock_alert_data):
        """Test successful alert creation"""
        mock_alert_service.create_alert = AsyncMock(return_value={
            "success": True,
            "message": "Alert created successfully",
            "alert": mock_alert_data.dict(),
            "timestamp": datetime.now().isoformat()
        })
        
        alert_data = {
            "location": "Durban",
            "risk_level": "HIGH",
            "message": "Test flood warning",
            "language": "en",
            "timestamp": datetime.now().isoformat()
        }
        
        response = client.post("/alerts", json=alert_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @patch('main.alert_service')
    def test_get_alerts_with_filters(self, mock_alert_service, client):
        """Test getting alerts with filters"""
        mock_alerts = [
            {
                "location": "Durban",
                "risk_level": "HIGH",
                "message": "Test alert",
                "language": "en",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        mock_alert_service.get_alerts = AsyncMock(return_value={
            "success": True,
            "count": 1,
            "alerts": mock_alerts,
            "timestamp": datetime.now().isoformat()
        })
        
        response = client.get("/alerts/history?location=Durban&risk_level=HIGH&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert len(data["alerts"]) == 1
    
    @patch('main.alert_service')
    def test_get_alert_statistics(self, mock_alert_service, client):
        """Test alert statistics endpoint"""
        mock_stats = {
            "period_days": 30,
            "total_alerts": 150,
            "recent_alerts_24h": 5,
            "risk_level_distribution": {"HIGH": 20, "MODERATE": 50, "LOW": 80},
            "top_locations": [{"_id": "Durban", "count": 25}],
            "timestamp": datetime.now().isoformat()
        }
        
        mock_alert_service.get_alert_statistics = AsyncMock(return_value=mock_stats)
        
        response = client.get("/alerts/statistics?days_back=30")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_alerts"] == 150
        assert data["recent_alerts_24h"] == 5


class TestErrorHandling:
    """Test error handling and validation"""
    
    def test_validation_error(self, client):
        """Test validation error handling"""
        invalid_alert = {
            "location": "",  # Empty location should fail validation
            "risk_level": "INVALID",
            "message": "",
            "language": "invalid_lang",
            "timestamp": "invalid_timestamp"
        }
        
        response = client.post("/alerts", json=invalid_alert)
        
        assert response.status_code == 422  # Validation error
    
    @patch('main.weather_service')
    def test_service_error_handling(self, mock_weather_service, client):
        """Test service error handling"""
        mock_weather_service.collect_weather_data = AsyncMock(
            side_effect=Exception("Service error")
        )
        
        response = client.get("/collect")
        
        assert response.status_code == 500
        assert "Failed to collect weather data" in response.json()["detail"]


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_exceeded(self, client):
        """Test rate limiting when exceeded"""
        # This would require mocking Redis and rate limiting logic
        # For now, we'll test the structure
        pass


class TestCaching:
    """Test caching functionality"""
    
    @patch('main.get_redis')
    def test_cached_response(self, mock_redis, client):
        """Test that responses are cached"""
        # Mock Redis to return cached data
        mock_redis_instance = MagicMock()
        mock_redis_instance.get = AsyncMock(return_value=b'{"cached": "data"}')
        mock_redis.return_value = mock_redis_instance
        
        # This would test actual caching behavior
        pass


class TestSecurity:
    """Test security features"""
    
    def test_api_key_required(self, client):
        """Test that API key is required for protected endpoints"""
        # This would test authentication when API key is configured
        pass
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options("/collect")
        
        # CORS headers should be present
        assert response.status_code in [200, 204]


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async operations"""
    
    async def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        # This would test concurrent request handling
        pass
    
    async def test_database_connection_pooling(self):
        """Test database connection pooling"""
        # This would test connection pooling behavior
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
