"""
NOAA/Open-Meteo Weather Data Fetcher

Fetches historical weather data using the Open-Meteo Archive API.
This API is free and does not require authentication.

Data includes: precipitation, temperature, wind, soil moisture, pressure.
"""

import httpx
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import logging
import os
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class NOAAFetcher:
    """Fetch historical weather data from Open-Meteo Archive API"""
    
    # Open-Meteo Archive API (free, no auth required)
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # Weather variables to fetch
    DAILY_VARIABLES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    ]
    
    # Key locations in Southern Africa for initial data collection
    KEY_LOCATIONS = [
        {"name": "Johannesburg", "lat": -26.2044, "lon": 28.0456, "country": "South Africa"},
        {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241, "country": "South Africa"},
        {"name": "Durban", "lat": -29.8587, "lon": 31.0218, "country": "South Africa"},
        {"name": "Harare", "lat": -17.8252, "lon": 31.0335, "country": "Zimbabwe"},
        {"name": "Gaborone", "lat": -24.6282, "lon": 25.9231, "country": "Botswana"},
        {"name": "Maputo", "lat": -25.9692, "lon": 32.5732, "country": "Mozambique"},
        {"name": "Lusaka", "lat": -15.3875, "lon": 28.3228, "country": "Zambia"},
        {"name": "Windhoek", "lat": -22.5609, "lon": 17.0658, "country": "Namibia"},
        {"name": "Lilongwe", "lat": -13.9626, "lon": 33.7741, "country": "Malawi"},
        {"name": "Luanda", "lat": -8.8368, "lon": 13.2343, "country": "Angola"},
    ]
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize weather fetcher
        
        Args:
            cache_dir: Directory for caching API responses
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/weather")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client
    
    async def fetch_weather_history(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily weather data for a location
        
        Args:
            latitude: Latitude (-90 to 90)
            longitude: Longitude (-180 to 180)
            start_date: Start date as YYYY-MM-DD
            end_date: End date as YYYY-MM-DD
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns:
            - date, latitude, longitude
            - temp_max, temp_min, temp_mean
            - precipitation, rain, snowfall, precipitation_hours
            - wind_speed_max, wind_gusts_max, wind_direction
            - radiation, evapotranspiration
        """
        logger.info(f"Fetching weather: ({latitude}, {longitude}) from {start_date} to {end_date}")
        
        # Check cache first
        cache_key = self._cache_key(latitude, longitude, start_date, end_date)
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info(f"  Using cached data ({len(cached)} records)")
                return cached
        
        try:
            client = await self._get_client()
            
            # Open-Meteo limits requests to ~2 years at a time
            all_data = []
            current_start = pd.to_datetime(start_date)
            final_end = pd.to_datetime(end_date)
            
            while current_start < final_end:
                # Chunk into 2-year periods
                current_end = min(current_start + timedelta(days=700), final_end)
                
                chunk = await self._fetch_chunk(
                    client, latitude, longitude,
                    current_start.strftime("%Y-%m-%d"),
                    current_end.strftime("%Y-%m-%d")
                )
                
                if chunk is not None and len(chunk) > 0:
                    all_data.append(chunk)
                
                current_start = current_end + timedelta(days=1)
                await asyncio.sleep(0.3)  # Rate limiting
            
            if not all_data:
                logger.warning("No weather data retrieved")
                return self._empty_dataframe()
            
            df = pd.concat(all_data, ignore_index=True)
            df = df.drop_duplicates(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            # Cache the result
            if use_cache:
                self._save_cache(cache_key, df)
            
            logger.info(f"  Retrieved {len(df)} weather records")
            return df
            
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}", exc_info=True)
            return self._empty_dataframe()
    
    async def _fetch_chunk(
        self,
        client: httpx.AsyncClient,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch a single chunk of weather data"""
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(self.DAILY_VARIABLES),
            "timezone": "auto",
        }
        
        try:
            response = await client.get(self.BASE_URL, params=params)
            
            if response.status_code == 429:
                logger.warning("Rate limit hit, waiting...")
                await asyncio.sleep(10)
                response = await client.get(self.BASE_URL, params=params)
            
            response.raise_for_status()
            data = response.json()
            
            return self._normalize_weather(data, latitude, longitude)
            
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Chunk fetch error: {e}")
            return None
    
    def _normalize_weather(
        self,
        raw_response: Dict[str, Any],
        latitude: float,
        longitude: float,
    ) -> pd.DataFrame:
        """Convert API response to standard schema"""
        
        daily = raw_response.get("daily", {})
        
        if not daily or "time" not in daily:
            return self._empty_dataframe()
        
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "latitude": latitude,
            "longitude": longitude,
            
            # Temperature
            "temp_max": daily.get("temperature_2m_max"),
            "temp_min": daily.get("temperature_2m_min"),
            "temp_mean": daily.get("temperature_2m_mean"),
            
            # Precipitation
            "precipitation": daily.get("precipitation_sum"),
            "rain": daily.get("rain_sum"),
            "snowfall": daily.get("snowfall_sum"),
            "precipitation_hours": daily.get("precipitation_hours"),
            
            # Wind
            "wind_speed_max": daily.get("wind_speed_10m_max"),
            "wind_gusts_max": daily.get("wind_gusts_10m_max"),
            "wind_direction": daily.get("wind_direction_10m_dominant"),
            
            # Radiation and ET
            "radiation": daily.get("shortwave_radiation_sum"),
            "evapotranspiration": daily.get("et0_fao_evapotranspiration"),
        })
        
        # Add metadata
        df["source"] = "open_meteo"
        df["fetched_at"] = datetime.now().isoformat()
        
        return df
    
    async def fetch_multiple_locations(
        self,
        locations: Optional[List[Dict[str, Any]]] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
    ) -> pd.DataFrame:
        """
        Fetch weather data for multiple locations
        
        Args:
            locations: List of dicts with lat, lon, name. Defaults to KEY_LOCATIONS.
            start_date: Start date
            end_date: End date
            
        Returns:
            Combined DataFrame for all locations
        """
        if locations is None:
            locations = self.KEY_LOCATIONS
        
        all_data = []
        
        for loc in locations:
            logger.info(f"Fetching weather for {loc.get('name', 'Unknown')}")
            
            df = await self.fetch_weather_history(
                latitude=loc["lat"],
                longitude=loc["lon"],
                start_date=start_date,
                end_date=end_date,
            )
            
            if len(df) > 0:
                df["location_name"] = loc.get("name", "")
                df["country"] = loc.get("country", "")
                all_data.append(df)
            
            await asyncio.sleep(0.5)  # Rate limiting between locations
        
        if not all_data:
            return self._empty_dataframe()
        
        return pd.concat(all_data, ignore_index=True)
    
    def _cache_key(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> str:
        """Generate cache key for request parameters"""
        key_str = f"{latitude:.4f}_{longitude:.4f}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load cached data if available and fresh"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if cache_file.exists():
            # Check if cache is less than 24 hours old
            age = datetime.now().timestamp() - cache_file.stat().st_mtime
            if age < 86400:  # 24 hours
                try:
                    return pd.read_parquet(cache_file)
                except Exception as e:
                    logger.warning(f"Cache read error: {e}")
        
        return None
    
    def _save_cache(self, cache_key: str, df: pd.DataFrame):
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        try:
            df.to_parquet(cache_file, index=False)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _empty_dataframe(self) -> pd.DataFrame:
        """Return empty DataFrame with expected columns"""
        return pd.DataFrame(columns=[
            "date", "latitude", "longitude",
            "temp_max", "temp_min", "temp_mean",
            "precipitation", "rain", "snowfall", "precipitation_hours",
            "wind_speed_max", "wind_gusts_max", "wind_direction",
            "radiation", "evapotranspiration",
            "source", "fetched_at"
        ])
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
