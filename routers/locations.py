from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import List, Optional
import structlog

from utils.dependencies import verify_api_key, rate_limit
from services.location_service import LocationService
from models.model import LocationCreate, LocationUpdate, LocationSearch, GeocodeRequest, LocationPreset

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/locations",
    tags=["Location Management"]
)

@router.post("", summary="Create new location")
async def create_location(
    request: Request,
    location_data: LocationCreate,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Create a new location with coordinates and metadata"""
    try:
        location_service: LocationService = request.app.state.location_service
        result = await location_service.create_location(location_data)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Location creation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create location: {str(e)}")

@router.get("/{location_id}", summary="Get location by ID")
async def get_location(
    request: Request,
    location_id: str,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Get location details by ID"""
    try:
        location_service: LocationService = request.app.state.location_service
        location = await location_service.get_location(location_id)
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
        return {"success": True, "location": location}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get location failed", location_id=location_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get location: {str(e)}")

@router.put("/{location_id}", summary="Update location")
async def update_location(
    request: Request,
    location_id: str,
    update_data: LocationUpdate,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Update location details"""
    try:
        location_service: LocationService = request.app.state.location_service
        result = await location_service.update_location(location_id, update_data)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Location update failed", location_id=location_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update location: {str(e)}")

@router.delete("/{location_id}", summary="Delete location")
async def delete_location(
    request: Request,
    location_id: str,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Soft delete a location"""
    try:
        location_service: LocationService = request.app.state.location_service
        result = await location_service.delete_location(location_id)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Location deletion failed", location_id=location_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete location: {str(e)}")

@router.post("/search", summary="Search locations")
async def search_locations(
    request: Request,
    search: LocationSearch,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Search and filter locations with advanced criteria"""
    try:
        location_service: LocationService = request.app.state.location_service
        result = await location_service.search_locations(search)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Location search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to search locations: {str(e)}")

@router.post("/geocode", summary="Geocode location name")
async def geocode_location(
    request: Request,
    req: GeocodeRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Find coordinates for a location name using geocoding"""
    try:
        location_service: LocationService = request.app.state.location_service
        result = await location_service.geocode_location(req)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Geocoding failed", query=req.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Geocoding failed: {str(e)}")

@router.post("/presets", summary="Create location preset")
async def create_location_preset(
    request: Request,
    preset_data: LocationPreset,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Create a preset collection of locations for easy access"""
    try:
        location_service: LocationService = request.app.state.location_service
        result = await location_service.create_preset(preset_data)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Preset creation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create preset: {str(e)}")

@router.get("/presets/{preset_id}", summary="Get preset locations")
async def get_preset_locations(
    request: Request,
    preset_id: str,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Get all locations in a preset"""
    try:
        location_service: LocationService = request.app.state.location_service
        result = await location_service.get_preset_locations(preset_id)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get preset failed", preset_id=preset_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get preset: {str(e)}")

@router.get("/initialize", summary="Initialize default locations")
async def initialize_locations(
    request: Request,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Initialize database with default South African locations"""
    try:
        location_service: LocationService = request.app.state.location_service
        result = await location_service.initialize_default_locations()
        return result
    except Exception as e:
        logger.error("Location initialization failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to initialize locations: {str(e)}")

@router.get("/coords", summary="Get location coordinates")
async def get_location_coordinates(
    request: Request,
    location_ids: Optional[List[str]] = Query(None, description="Specific location IDs to get coordinates for"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Get coordinates for locations (compatible with prediction system)"""
    try:
        location_service: LocationService = request.app.state.location_service
        coords = await location_service.get_locations_coords(location_ids)
        return {
            "success": True,
            "coordinates": coords,
            "count": len(coords)
        }
    except Exception as e:
        logger.error("Get coordinates failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get coordinates: {str(e)}")
