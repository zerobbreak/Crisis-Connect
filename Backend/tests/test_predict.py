import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from services.predict import (
    fetch_weather_and_marine_data,
    collect_all_data,
    generate_risk_scores,
    calculate_household_resources,
    DISTRICT_COORDS
)

class TestPredictionService:
    @patch('services.predict.openmeteo')
    def test_fetch_weather_and_marine_data(self, mock_openmeteo):
        """Test fetching weather and marine data"""
        # Mock the weather API response
        mock_weather_hourly = MagicMock()
        mock_weather_response = MagicMock()
        mock_weather_response.Hourly.return_value = mock_weather_hourly
        
        # Mock the marine API response for coastal locations
        mock_marine_hourly = MagicMock()
        mock_marine_response = MagicMock()
        mock_marine_response.Hourly.return_value = mock_marine_hourly
        
        # Configure the mock to return different responses based on is_coastal
        mock_openmeteo.weather_api.side_effect = [
            [mock_weather_response],  # First call for weather
            [mock_marine_response]    # Second call for marine (if coastal)
        ]
        
        # Test for coastal location
        lat, lon = -29.8587, 31.0218  # Durban coordinates
        weather_hourly, marine_hourly = fetch_weather_and_marine_data(lat, lon, is_coastal=True)
        
        assert weather_hourly == mock_weather_hourly
        assert marine_hourly == mock_marine_hourly
        assert mock_openmeteo.weather_api.call_count == 2
        
        # Reset mock for non-coastal test
        mock_openmeteo.reset_mock()
        mock_openmeteo.weather_api.side_effect = [[mock_weather_response]]
        
        # Test for non-coastal location
        weather_hourly, marine_hourly = fetch_weather_and_marine_data(lat, lon, is_coastal=False)
        
        assert weather_hourly == mock_weather_hourly
        assert marine_hourly is None
        assert mock_openmeteo.weather_api.call_count == 1
    
    @patch('services.predict.fetch_weather_and_marine_data')
    @patch('services.predict._geocode')
    def test_collect_all_data_with_place_name(self, mock_geocode, mock_fetch_data):
        """Test collecting data with place name"""
        # Mock geocoding response
        mock_location = MagicMock()
        mock_location.latitude = -29.8587
        mock_location.longitude = 31.0218
        mock_geocode.return_value = mock_location
        
        # Mock weather and marine data
        mock_weather = MagicMock()
        mock_weather.Variables.return_value = MagicMock()
        mock_marine = MagicMock()
        mock_marine.Variables.return_value = MagicMock()
        mock_fetch_data.return_value = (mock_weather, mock_marine)
        
        # Test with place name
        result = collect_all_data(["Durban"])
        
        assert result is not None
        mock_geocode.assert_called_once_with("Durban")
        mock_fetch_data.assert_called_once()
    
    @patch('services.predict.fetch_weather_and_marine_data')
    def test_collect_all_data_with_district(self, mock_fetch_data):
        """Test collecting data with district name"""
        # Mock weather and marine data
        mock_weather = MagicMock()
        mock_weather.Variables.return_value = MagicMock()
        mock_marine = MagicMock()
        mock_marine.Variables.return_value = MagicMock()
        mock_fetch_data.return_value = (mock_weather, mock_marine)
        
        # Test with district name using the district coordinates dictionary
        district_dict = {"eThekwini (Durban)": DISTRICT_COORDS["eThekwini (Durban)"]}
        result = collect_all_data(district_dict)
        
        assert result is not None
        # Should use coordinates from DISTRICT_COORDS
        mock_fetch_data.assert_called_once()
    
    @patch('services.predict.fetch_weather_and_marine_data')
    def test_collect_all_data_with_coordinates(self, mock_fetch_data):
        """Test collecting data with explicit coordinates"""
        # Mock weather and marine data
        mock_weather = MagicMock()
        mock_weather.Variables.return_value = MagicMock()
        mock_marine = MagicMock()
        mock_marine.Variables.return_value = MagicMock()
        mock_fetch_data.return_value = (mock_weather, mock_marine)
        
        # Test with explicit coordinates
        lat, lon = -29.8587, 31.0218
        result = collect_all_data([(-29.8587, 31.0218)])
        
        assert result is not None
        mock_fetch_data.assert_called_once()
    
    def test_generate_risk_scores(self):
        """Test generating risk scores from weather data"""
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.25, 0.75]])  # 75% probability of high risk
        
        # Create sample weather data
        weather_data = pd.DataFrame({
            'location': ['Test Location'],
            'lat': [-29.8587],
            'lon': [31.0218],
            'temp_c': [25.5],
            'humidity': [65.0],
            'wind_kph': [15.2],
            'pressure_mb': [1013.0],
            'precip_mm': [2.5],
            'cloud': [20.0],
            'wave_height': [1.2],
            'is_severe': [1],
            'anomaly_score': [50.0]
        })
        
        # Generate risk scores
        result = generate_risk_scores(mock_model, weather_data)
        
        # Verify results
        assert len(result) == 1
        assert 'composite_risk_score' in result.columns
        assert 'risk_category' in result.columns
        assert 'model_risk_score' in result.columns
        assert result['model_risk_score'].iloc[0] == 75.0  # From mock probability
        assert result['risk_category'].iloc[0] == 'High'  # Based on threshold
    
    def test_calculate_household_resources(self):
        """Test calculating household resources based on risk severity"""
        # Test high risk resources
        high_resources = calculate_household_resources('High', household_size=4)
        assert high_resources['food_packs'] > 0
        assert high_resources['water_gallons'] > 0
        assert high_resources['shelter_needed'] is True
        assert high_resources['boats_needed'] > 0
        
        # Test medium risk resources
        medium_resources = calculate_household_resources('Medium', household_size=4)
        assert medium_resources['food_packs'] > 0
        assert medium_resources['water_gallons'] > 0
        assert medium_resources['shelter_needed'] is True
        assert medium_resources['boats_needed'] == 0
        
        # Test low risk resources
        low_resources = calculate_household_resources('Low', household_size=4)
        assert low_resources['food_packs'] >= 0
        assert low_resources['water_gallons'] >= 0
        assert low_resources['shelter_needed'] is False
        assert low_resources['boats_needed'] == 0