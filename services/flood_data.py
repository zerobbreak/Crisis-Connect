"""
Flood Data Service
Handles fetching river discharge and elevation data from Open-Meteo APIs.
"""

import httpx
import logging
from typing import Dict, Optional, Tuple
import asyncio

logger = logging.getLogger("crisisconnect.flood_data")

class FloodDataService:
    """
    Service for fetching flood-related data (River Discharge, Elevation).
    Uses Open-Meteo Flood API and Elevation API.
    """
    
    def __init__(self):
        self.flood_api_url = "https://flood-api.open-meteo.com/v1/flood"
        self.elevation_api_url = "https://api.open-meteo.com/v1/elevation"
        self.client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def fetch_river_discharge(
        self, 
        lat: float, 
        lon: float, 
        days_forecast: int = 7
    ) -> Optional[Dict]:
        """
        Fetch river discharge data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            days_forecast: Number of days to forecast (max 92)
            
        Returns:
            Dictionary with river discharge data or None if failed
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "river_discharge,river_discharge_mean,river_discharge_median",
            "forecast_days": days_forecast
        }
        
        try:
            response = await self.client.get(self.flood_api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched river discharge for {lat}, {lon}")
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch river discharge: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in fetch_river_discharge: {e}")
            return None

    async def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """
        Get precise elevation for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Elevation in meters or None if failed
        """
        params = {
            "latitude": lat,
            "longitude": lon
        }
        
        try:
            response = await self.client.get(self.elevation_api_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "elevation" in data and data["elevation"]:
                elevation = data["elevation"][0]
                return elevation
            return None
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch elevation: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in get_elevation: {e}")
            return None

    async def get_flood_risk_factors(self, lat: float, lon: float) -> Dict:
        """
        Get combined flood risk factors (Discharge + Elevation).
        
        Returns:
            Dict with 'river_discharge_max', 'elevation', 'is_low_lying'
        """
        discharge_task = self.fetch_river_discharge(lat, lon, days_forecast=3)
        elevation_task = self.get_elevation(lat, lon)
        
        discharge_data, elevation = await asyncio.gather(discharge_task, elevation_task)
        
        result = {
            "river_discharge_max": 0.0,
            "river_discharge_forecast": [],
            "elevation": elevation,
            "is_low_lying": False
        }
        
        if discharge_data and "daily" in discharge_data:
            daily = discharge_data["daily"]
            if "river_discharge" in daily and daily["river_discharge"]:
                # Filter out None values
                valid_discharge = [d for d in daily["river_discharge"] if d is not None]
                if valid_discharge:
                    result["river_discharge_max"] = max(valid_discharge)
                    result["river_discharge_forecast"] = valid_discharge
        
        if elevation is not None:
            # Simple heuristic: Lower elevation = higher risk (relative to surroundings would be better, but absolute is a start)
            # This is a very rough approximation. Ideally, we'd compare to river bank elevation.
            # For now, we just return the value.
            pass
            
        return result

# Singleton instance
flood_service = FloodDataService()
