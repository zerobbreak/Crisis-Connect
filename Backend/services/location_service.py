"""
Location Management Service
Handles dynamic location CRUD operations, presets, and geocoding
"""
import structlog
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
import httpx
from models.model import Location, LocationCreate, LocationUpdate, LocationPreset, LocationSearch, GeocodeRequest
from services.predict import DISTRICT_COORDS
import asyncio

logger = structlog.get_logger(__name__)

class LocationService:
    """Service for managing dynamic locations"""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection: AsyncIOMotorCollection = db.locations
        self.presets_collection: AsyncIOMotorCollection = db.location_presets

    async def initialize_default_locations(self) -> Dict[str, Any]:
        """Initialize database with default South African locations"""
        try:
            # Check if locations already exist
            count = await self.collection.count_documents({})
            if count > 0:
                return {"success": True, "message": f"Locations already initialized ({count} locations)"}

            # Convert hardcoded locations to database format
            locations_to_insert = []
            for name, (lat, lon) in DISTRICT_COORDS.items():
                # Extract region info from name if possible
                region = None
                if "Cape Town" in name or "George" in name or "Mossel Bay" in name or "Hermanus" in name or "Saldanha" in name or "Knysna" in name:
                    region = "Western Cape"
                elif "Gqeberha" in name or "East London" in name or "Mthatha" in name:
                    region = "Eastern Cape"
                elif "eThekwini" in name or "King Cetshwayo" in name or "Ugu" in name or "iLembe" in name or "uThukela" in name or "uMgungundlovu" in name or "Amajuba" in name or "Uthungulu" in name or "UMkhanyakude" in name or "Zululand" in name:
                    region = "KwaZulu-Natal"
                elif "Johannesburg" in name or "Pretoria" in name or "Bloemfontein" in name or "Polokwane" in name or "Mbombela" in name or "Rustenburg" in name or "Kimberley" in name:
                    if "Johannesburg" in name or "Pretoria" in name or "Rustenburg" in name:
                        region = "Gauteng"
                    elif "Bloemfontein" in name:
                        region = "Free State"
                    elif "Polokwane" in name:
                        region = "Limpopo"
                    elif "Mbombela" in name:
                        region = "Mpumalanga"
                    elif "Kimberley" in name:
                        region = "Northern Cape"

                is_coastal = any(keyword in name.lower() for keyword in [
                    "durban", "richards bay", "port shepstone", "ballito", "cape town",
                    "george", "mossel bay", "hermanus", "saldanha", "knysna",
                    "port elizabeth", "east london", "mthatha", "richards bay",
                    "mtubatuba", "empangeni", "ballito"
                ])

                location = Location(
                    name=name,
                    display_name=name,
                    latitude=lat,
                    longitude=lon,
                    country="South Africa",
                    region=region,
                    district=name,
                    is_coastal=is_coastal,
                    tags=["default", "south-africa"],
                    metadata={"source": "hardcoded_defaults"}
                )
                locations_to_insert.append(location.model_dump())

            if locations_to_insert:
                result = await self.collection.insert_many(locations_to_insert)
                return {
                    "success": True,
                    "message": f"Initialized {len(result.inserted_ids)} default locations",
                    "count": len(result.inserted_ids)
                }

            return {"success": True, "message": "No locations to initialize"}

        except Exception as e:
            logger.error("Failed to initialize default locations", error=str(e))
            return {"success": False, "message": f"Failed to initialize locations: {str(e)}"}

    async def create_location(self, location_data: LocationCreate) -> Dict[str, Any]:
        """Create a new location"""
        try:
            # Check for duplicate name
            existing = await self.collection.find_one({"name": location_data.name, "is_active": True})
            if existing:
                return {"success": False, "message": f"Location '{location_data.name}' already exists"}

            location = Location(**location_data.model_dump())
            result = await self.collection.insert_one(location.model_dump())

            logger.info("Location created", location_id=result.inserted_id, name=location.name)
            return {
                "success": True,
                "message": "Location created successfully",
                "location_id": str(result.inserted_id),
                "location": location.model_dump()
            }

        except Exception as e:
            logger.error("Failed to create location", error=str(e))
            return {"success": False, "message": f"Failed to create location: {str(e)}"}

    async def get_location(self, location_id: str) -> Optional[Dict[str, Any]]:
        """Get location by ID"""
        try:
            location = await self.collection.find_one({"_id": location_id, "is_active": True})
            return location
        except Exception as e:
            logger.error("Failed to get location", location_id=location_id, error=str(e))
            return None

    async def update_location(self, location_id: str, update_data: LocationUpdate) -> Dict[str, Any]:
        """Update location"""
        try:
            # Check if location exists
            existing = await self.collection.find_one({"_id": location_id, "is_active": True})
            if not existing:
                return {"success": False, "message": "Location not found"}

            update_dict = update_data.model_dump(exclude_unset=True)
            update_dict["updated_at"] = datetime.utcnow()

            result = await self.collection.update_one(
                {"_id": location_id},
                {"$set": update_dict}
            )

            if result.modified_count > 0:
                logger.info("Location updated", location_id=location_id)
                return {"success": True, "message": "Location updated successfully"}
            else:
                return {"success": False, "message": "No changes made to location"}

        except Exception as e:
            logger.error("Failed to update location", location_id=location_id, error=str(e))
            return {"success": False, "message": f"Failed to update location: {str(e)}"}

    async def delete_location(self, location_id: str) -> Dict[str, Any]:
        """Soft delete location"""
        try:
            result = await self.collection.update_one(
                {"_id": location_id, "is_active": True},
                {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
            )

            if result.modified_count > 0:
                logger.info("Location deleted", location_id=location_id)
                return {"success": True, "message": "Location deleted successfully"}
            else:
                return {"success": False, "message": "Location not found"}

        except Exception as e:
            logger.error("Failed to delete location", location_id=location_id, error=str(e))
            return {"success": False, "message": f"Failed to delete location: {str(e)}"}

    async def search_locations(self, search: LocationSearch) -> Dict[str, Any]:
        """Search and filter locations"""
        try:
            query = {"is_active": search.is_active} if search.is_active is not None else {}

            if search.name:
                query["name"] = {"$regex": search.name, "$options": "i"}
            if search.country:
                query["country"] = {"$regex": search.country, "$options": "i"}
            if search.region:
                query["region"] = {"$regex": search.region, "$options": "i"}
            if search.district:
                query["district"] = {"$regex": search.district, "$options": "i"}
            if search.is_coastal is not None:
                query["is_coastal"] = search.is_coastal
            if search.tags:
                query["tags"] = {"$in": search.tags}

            # Geographic bounds
            if any([search.latitude_min, search.latitude_max, search.longitude_min, search.longitude_max]):
                geo_query = {}
                if search.latitude_min is not None or search.latitude_max is not None:
                    geo_query["latitude"] = {}
                    if search.latitude_min is not None:
                        geo_query["latitude"]["$gte"] = search.latitude_min
                    if search.latitude_max is not None:
                        geo_query["latitude"]["$lte"] = search.latitude_max

                if search.longitude_min is not None or search.longitude_max is not None:
                    geo_query["longitude"] = {}
                    if search.longitude_min is not None:
                        geo_query["longitude"]["$gte"] = search.longitude_min
                    if search.longitude_max is not None:
                        geo_query["longitude"]["$lte"] = search.longitude_max

                if geo_query:
                    query.update(geo_query)

            # Get results
            cursor = self.collection.find(query).skip(search.offset).limit(search.limit)
            locations = await cursor.to_list(length=None)

            # Get total count
            total_count = await self.collection.count_documents(query)

            return {
                "success": True,
                "locations": locations,
                "total_count": total_count,
                "limit": search.limit,
                "offset": search.offset
            }

        except Exception as e:
            logger.error("Failed to search locations", error=str(e))
            return {"success": False, "message": f"Failed to search locations: {str(e)}"}

    async def geocode_location(self, request: GeocodeRequest) -> Dict[str, Any]:
        """Geocode location name to coordinates using external API"""
        try:
            # Using Nominatim (OpenStreetMap) geocoding API
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": request.query,
                "country": request.country,
                "format": "json",
                "limit": request.limit,
                "addressdetails": 1
            }

            headers = {
                "User-Agent": "CrisisConnect/1.0"
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params, headers=headers)

                if response.status_code != 200:
                    return {"success": False, "message": "Geocoding service unavailable"}

                results = response.json()

                if not results:
                    return {"success": False, "message": "No locations found for the given query"}

                # Format results
                locations = []
                for result in results:
                    locations.append({
                        "name": result.get("display_name", "").split(",")[0],
                        "display_name": result.get("display_name", ""),
                        "latitude": float(result["lat"]),
                        "longitude": float(result["lon"]),
                        "country": result.get("address", {}).get("country", ""),
                        "region": result.get("address", {}).get("state", ""),
                        "district": result.get("address", {}).get("county", ""),
                        "importance": float(result.get("importance", 0))
                    })

                return {
                    "success": True,
                    "query": request.query,
                    "locations": locations
                }

        except Exception as e:
            logger.error("Geocoding failed", query=request.query, error=str(e))
            return {"success": False, "message": f"Geocoding failed: {str(e)}"}

    async def create_preset(self, preset_data: LocationPreset) -> Dict[str, Any]:
        """Create a location preset"""
        try:
            result = await self.presets_collection.insert_one(preset_data.model_dump())
            logger.info("Location preset created", preset_id=result.inserted_id, name=preset_data.name)
            return {
                "success": True,
                "message": "Preset created successfully",
                "preset_id": str(result.inserted_id)
            }
        except Exception as e:
            logger.error("Failed to create preset", error=str(e))
            return {"success": False, "message": f"Failed to create preset: {str(e)}"}

    async def get_preset_locations(self, preset_id: str) -> Dict[str, Any]:
        """Get locations for a preset"""
        try:
            preset = await self.presets_collection.find_one({"_id": preset_id})
            if not preset:
                return {"success": False, "message": "Preset not found"}

            # Get locations by IDs
            location_ids = preset["locations"]
            locations = await self.collection.find({
                "_id": {"$in": location_ids},
                "is_active": True
            }).to_list(length=None)

            return {
                "success": True,
                "preset": preset,
                "locations": locations
            }

        except Exception as e:
            logger.error("Failed to get preset locations", preset_id=preset_id, error=str(e))
            return {"success": False, "message": f"Failed to get preset locations: {str(e)}"}

    async def get_locations_coords(self, location_ids: List[str] = None) -> Dict[str, Tuple[float, float]]:
        """Get coordinates for locations (compatible with existing predict service)"""
        try:
            if location_ids:
                query = {"_id": {"$in": location_ids}, "is_active": True}
            else:
                query = {"is_active": True}

            locations = await self.collection.find(query).to_list(length=None)

            coords = {}
            for loc in locations:
                coords[loc["name"]] = (loc["latitude"], loc["longitude"])

            return coords

        except Exception as e:
            logger.error("Failed to get location coordinates", error=str(e))
            return {}
