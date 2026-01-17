import sys
import os
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.predict import compare_with_historical_disasters

def test_disaster_match():
    """
    Test that weather matching a known disaster triggers high risk.
    """
    # 1. Create a weather pattern similar to 2022 KZN Floods (High Precip)
    disaster_weather = {
        "precip_mm": 280.0, # Very close to 300mm threshold
        "wind_kph": 35.0,
        "humidity": 85.0
    }
    
    risk_score = compare_with_historical_disasters(disaster_weather, "Durban")
    
    print(f"Disaster Weather Risk Score: {risk_score}")
    assert risk_score >= 90.0, "Should detect high risk for disaster-like weather"

def test_normal_weather():
    """
    Test that normal weather does not trigger disaster match.
    """
    normal_weather = {
        "precip_mm": 5.0,
        "wind_kph": 10.0,
        "humidity": 50.0
    }
    
    risk_score = compare_with_historical_disasters(normal_weather, "Durban")
    
    print(f"Normal Weather Risk Score: {risk_score}")
    assert risk_score == 0.0, "Should not detect risk for normal weather"

if __name__ == "__main__":
    test_disaster_match()
    test_normal_weather()
    print("✅ All historical comparison tests passed!")
