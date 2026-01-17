import logging
import httpx
from datetime import datetime

logger = logging.getLogger("crisisconnect.weather")


async def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch current weather data from Open-Meteo.

    Parameters
    ----------
    lat: float
        Latitude of the location.
    lon: float
        Longitude of the location.

    Returns
    -------
    dict
        Parsed JSON response containing temperature, precipitation, wind, etc.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"Fetched weather for {lat},{lon}")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch weather data for {lat},{lon}: {e}")
            return {}
