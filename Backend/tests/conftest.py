import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from main import app    

from utils.db import get_db

@pytest.fixture
def test_client():
    """Return a TestClient instance for FastAP[pytest]
pythonpath = .
addopts = -vI app testing"""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_db():
    """Create a mock MongoDB database for testing"""
    mock_db = MagicMock(spec=AsyncIOMotorDatabase)
    
    # Mock collections
    mock_db.weather_data = MagicMock()
    mock_db.predictions = MagicMock()
    mock_db.alerts = MagicMock()
    mock_db.historical_events = MagicMock()
    mock_db.historical_summary = MagicMock()
    mock_db.location_risks = MagicMock()
    
    return mock_db

@pytest.fixture
def mock_app_with_db(mock_db):
    """Patch the app to use the mock database"""
    with patch.object(app.state, 'db', mock_db):
        yield app

@pytest.fixture
def sample_weather_data():
    """Return sample weather data for testing"""
    return pd.DataFrame({
        'location': ['Test Location', 'Another Location'],
        'latitude': [-29.8587, -33.9249],
        'longitude': [31.0218, 18.4241],
        'temperature': [25.5, 22.3],
        'humidity': [65.0, 70.5],
        'rainfall': [2.5, 0.0],
        'wind_speed': [15.2, 10.8],
        'wave_height': [1.2, 0.5],
        'timestamp': ['2023-10-15 12:00:00', '2023-10-15 12:00:00']
    })

@pytest.fixture
def sample_historical_data():
    """Return sample historical disaster data for testing"""
    return pd.DataFrame({
        'location': ['Test Location', 'Another Location'],
        'latitude': [-29.8587, -33.9249],
        'longitude': [31.0218, 18.4241],
        'total_deaths': [120, 5],
        'severity': ['High', 'Low'],
        'timestamp': ['2022-01-15', '2022-02-20']
    })

@pytest.fixture
def sample_prediction_data():
    """Return sample prediction data for testing"""
    return pd.DataFrame({
        'location': ['Test Location', 'Another Location'],
        'composite_risk_score': [75.5, 35.2],
        'risk_category': ['High', 'Low'],
        'wave_height': [1.2, 0.5],
        'household_resources': [
            {'food_packs': 12, 'water_gallons': 24, 'shelter_needed': True, 'boats_needed': 2},
            {'food_packs': 4, 'water_gallons': 8, 'shelter_needed': False, 'boats_needed': 0}
        ],
        'timestamp': ['2023-10-15 12:00:00', '2023-10-15 12:00:00']
    })