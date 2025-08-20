import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import json
from main import app, load_historical_data, load_weather_data
from models.model import LocationRequest

@pytest.fixture
def client():
    return TestClient(app)

class TestAPIEndpoints:
    @patch('main.load_historical_data')
    def test_get_historical_data(self, mock_load_data, client):
        """Test GET /historical endpoint"""
        # Mock the data loading function
        mock_df = pd.DataFrame({
            'location': ['Test Location'],
            'latitude': [-29.8587],
            'longitude': [31.0218],
            'total_deaths': [120],
            'severity': ['High']
        })
        mock_load_data.return_value = mock_df
        
        # Make request
        response = client.get("/historical")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['location'] == 'Test Location'
        assert data[0]['severity'] == 'High'
    
    @patch('main.load_weather_data')
    def test_get_weather_data(self, mock_load_data, client):
        """Test GET /weather endpoint"""
        # Mock the data loading function
        mock_df = pd.DataFrame({
            'location': ['Test Location'],
            'latitude': [-29.8587],
            'longitude': [31.0218],
            'temperature': [25.5],
            'humidity': [65.0],
            'rainfall': [2.5],
            'wind_speed': [15.2],
            'wave_height': [1.2],
            'timestamp': ['2023-10-15 12:00:00']
        })
        mock_load_data.return_value = mock_df
        
        # Make request
        response = client.get("/weather")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['location'] == 'Test Location'
        assert data[0]['temperature'] == 25.5
    
    @patch('main.collect_all_data')
    @patch('main.generate_risk_scores')
    @patch('main.calculate_household_resources')
    def test_predict_risk(self, mock_calculate_resources, mock_generate_scores, mock_collect_data, client):
        """Test POST /predict endpoint"""
        # Mock the prediction pipeline
        mock_weather_data = pd.DataFrame({
            'location': ['Durban'],
            'lat': [-29.8587],
            'lon': [31.0218],
            'temp_c': [25.5],
            'humidity': [65.0],
            'wind_kph': [15.2],
            'pressure_mb': [1013.0],
            'precip_mm': [2.5],
            'cloud': [20.0],
            'wave_height': [1.2]
        })
        mock_collect_data.return_value = mock_weather_data
        
        mock_risk_data = mock_weather_data.copy()
        mock_risk_data['composite_risk_score'] = [75.0]
        mock_risk_data['risk_category'] = ['High']
        mock_generate_scores.return_value = mock_risk_data
        
        mock_resource_data = mock_risk_data.copy()
        mock_resource_data['household_resources'] = [{
            'food_packs': 12,
            'water_gallons': 24,
            'shelter_needed': True,
            'boats_needed': 2
        }]
        mock_calculate_resources.return_value = mock_resource_data
        
        # Make request
        request_data = {
            "place_name": "Durban",
            "is_coastal": True
        }
        response = client.post("/predict", json=request_data)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data['location'] == 'Durban'
        assert data['composite_risk_score'] == 75.0
        assert data['risk_category'] == 'High'
        assert 'household_resources' in data
    
    @patch('main.generate_alerts_from_db')
    def test_get_alerts(self, mock_generate_alerts, client):
        """Test GET /alerts endpoint"""
        # Mock the alert generation function
        mock_generate_alerts.return_value = [
            {
                'location': 'Test Location',
                'risk_level': 'high',
                'message': 'Test alert message',
                'language': 'en',
                'timestamp': '2023-10-15 12:00:00'
            }
        ]
        
        # Make request
        response = client.get("/alerts")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['location'] == 'Test Location'
        assert data[0]['risk_level'] == 'high'
    
    def test_health_check(self, client):
        """Test GET /health endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "status" in response.json()