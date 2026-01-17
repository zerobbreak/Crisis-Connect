from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import structlog
from datetime import datetime

from utils.dependencies import verify_api_key, rate_limit, cache_response
from services.weather_service import WeatherService
from models.model import WeatherEntry

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Weather & Risk"]
)

class CollectRequest(BaseModel):
    locations: Optional[List[Dict[str, Any]]] = None
    location_ids: Optional[List[str]] = None

    class Config:
        schema_extra = {
            "example": {
                "locations": [
                    {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241},
                    {"name": "Johannesburg", "lat": -26.2041, "lon": 28.0473}
                ],
                "location_ids": ["64f1a2b3c4d5e6f7g8h9i0j1", "64f1a2b3c4d5e6f7g8h9i0j2"]
            }
        }

@router.get("/weather/collect", summary="Collect weather data for all active locations", tags=["Data Collection"])
async def collect_weather_data(
    request: Request,
    location_ids: Optional[List[str]] = Query(None, description="Specific location IDs to collect data for"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Collect real-time weather data for active locations or specified location IDs"""
    try:
        weather_service: WeatherService = request.app.state.weather_service
        result = await weather_service.collect_weather_data(location_ids=location_ids)

        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Weather data collection failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to collect weather data: {str(e)}"
        )

@router.post("/weather/collect", summary="Collect weather data for custom locations", tags=["Data Collection"])
async def collect_weather_data_custom(
    request: Request,
    payload: CollectRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Collect weather data for custom locations or location IDs"""
    try:
        weather_service: WeatherService = request.app.state.weather_service
        
        # Validate input
        if not payload.locations and not payload.location_ids:
            raise HTTPException(status_code=400, detail="Either locations or location_ids must be provided")

        # Validate locations if provided
        validated_locations = None
        if payload.locations:
            if len(payload.locations) > 100:
                raise HTTPException(status_code=400, detail="Too many locations. Maximum 100 allowed.")

            validated_locations = []
            for i, loc in enumerate(payload.locations):
                if not isinstance(loc, dict) or "lat" not in loc or "lon" not in loc:
                    continue
                try:
                    lat, lon = float(loc["lat"]), float(loc["lon"])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        validated_locations.append({
                            "name": loc.get("name", f"Custom_{i}"),
                            "lat": lat,
                            "lon": lon
                        })
                except (ValueError, TypeError):
                    continue

            if not validated_locations and not payload.location_ids:
                raise HTTPException(status_code=400, detail="No valid locations provided")

        result = await weather_service.collect_weather_data(
            locations=validated_locations,
            location_ids=payload.location_ids
        )
        
        if result["success"]:
            if result["count"] > 0 and result["count"] <= 50:
                sample_records = await weather_service.get_weather_data(limit=result["count"])
                result["records"] = sample_records
            else:
                result["records"] = []
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Custom weather data collection failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to collect custom weather data: {str(e)}")

@router.get("/risk/assess", summary="Get latest risk scores for all locations", tags=["Risk Assessment"])
@cache_response(expire_seconds=300)
async def assess_risk_all(request: Request):
    """Get the latest risk assessment data for all locations"""
    from utils.db import get_db
    db = get_db(request.app)
    
    try:
        predictions = await db["predictions"].find(
            {"composite_risk_score": {"$exists": True, "$ne": None}},
            {"_id": 0}
        ).sort("timestamp", -1).to_list(length=10000)

        if not predictions:
            return []

        # Group by location to get latest for each
        latest_predictions = {}
        for pred in predictions:
            location = pred.get("location")
            if location and location not in latest_predictions:
                latest_predictions[location] = pred

        return list(latest_predictions.values())

    except Exception as e:
        logger.error("Failed to fetch risk assessment data", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve risk assessments")

@router.post("/risk/predict", summary="Generate risk predictions using ML models", tags=["Risk Assessment"])
async def predict_risk_scores(
    request: Request,
    generate_alerts: bool = Query(False, description="Whether to generate alerts for high-risk locations"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Process available weather data and generate risk assessments using ML models"""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    try:
        weather_service: WeatherService = request.app.state.weather_service
        result = await weather_service.process_risk_assessment(model, generate_alerts)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Risk assessment processing failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process risk assessment: {str(e)}")

@router.get("/weather/current/{location_id}", summary="Get latest weather for location")
async def get_current_weather_by_id(
    request: Request,
    location_id: str,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Get the latest stored weather data for a specific location ID"""
    try:
        db = request.app.state.weather_service.db
        from bson import ObjectId
        
        # Find location first to get name
        location = await db["locations"].find_one({"_id": ObjectId(location_id)})
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
            
        # Get latest weather for this location name
        weather = await db["weather_data"].find_one(
            {"location": location["name"]},
            sort=[("timestamp", -1)]
        )
        
        if not weather:
            raise HTTPException(status_code=404, detail="No weather data found for this location")
            
        # Convert _id to string
        weather["_id"] = str(weather["_id"])
        
        return {
            "success": True,
            "location": location,
            "weather": weather
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get weather by ID", location_id=location_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get weather data: {str(e)}")
