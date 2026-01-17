import logging
import asyncio
from typing import List, Dict

from services.flood_data import flood_service
from services.weather_data import fetch_weather

logger = logging.getLogger("crisisconnect.data_ingestion")

class DataIngestionService:
    """Fetch real‑time weather and flood data for a list of locations.

    The service returns a dict mapping location IDs to a combined payload:
    {
        "weather": {...},
        "flood": {...}
    }
    """

    async def fetch_for_location(self, location: Dict) -> Dict:
        lat = location.get("latitude")
        lon = location.get("longitude")
        if lat is None or lon is None:
            logger.warning("Location missing coordinates", location=location)
            return {}
        try:
            weather_task = fetch_weather(lat, lon)
            flood_task = flood_service.fetch_river_discharge(lat, lon, days_forecast=1)
            weather, flood = await asyncio.gather(weather_task, flood_task)
            return {"weather": weather, "flood": flood}
        except Exception as e:
            logger.error(
                "Failed to ingest data for location",
                location_id=location.get("_id"),
                error=str(e),
                exc_info=True,
            )
            return {}

    async def fetch_all(self, locations: List[Dict]) -> Dict:
        """Fetch data for all provided locations concurrently.

        Returns a dict keyed by location ID.
        """
        results: Dict = {}
        tasks = []
        for loc in locations:
            loc_id = str(loc.get("_id"))
            tasks.append(self.fetch_for_location(loc))
            # store placeholder for ordering
            results[loc_id] = {}
        fetched = await asyncio.gather(*tasks)
        for loc, data in zip(locations, fetched):
            results[str(loc.get("_id"))] = data
        logger.info("Data ingestion completed for %d locations", len(locations))
        return results
