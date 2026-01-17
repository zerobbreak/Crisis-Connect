"""
Forecast API Router
Endpoints for time series flood risk forecasting
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import List, Optional, Dict
from pydantic import BaseModel
import structlog
from datetime import datetime

from utils.dependencies import verify_api_key, rate_limit
from services.forecast import ForecastService

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/forecast",
    tags=["Forecasting"]
)


class BatchForecastRequest(BaseModel):
    """Request model for batch forecast generation"""
    location_ids: List[str]
    horizons: Optional[List[int]] = [24, 48, 72]
    
    class Config:
        schema_extra = {
            "example": {
                "location_ids": ["64f1a2b3c4d5e6f7g8h9i0j1", "64f1a2b3c4d5e6f7g8h9i0j2"],
                "horizons": [24, 48, 72]
            }
        }


@router.get("/{location_id}", summary="Get multi-horizon forecast for location")
async def get_forecast(
    request: Request,
    location_id: str,
    horizons: Optional[str] = Query("24,48,72", description="Comma-separated forecast horizons in hours"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Generate multi-horizon flood risk forecast for a specific location.
    
    Returns predictions for 24h, 48h, and 72h ahead with confidence intervals,
    trend indicators, and early warning flags.
    """
    try:
        from utils.db import get_db
        from bson import ObjectId
        
        db = get_db(request.app)
        
        # Get location details
        try:
            location = await db["locations"].find_one({"_id": ObjectId(location_id)})
        except:
            raise HTTPException(status_code=400, detail="Invalid location ID format")
        
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
        
        # Parse horizons
        try:
            horizon_list = [int(h.strip()) for h in horizons.split(",")]
        except:
            horizon_list = [24, 48, 72]
        
        # Get forecast service
        forecast_service = getattr(request.app.state, 'forecast_service', None)
        if forecast_service is None:
            # Create forecast service on-the-fly
            from services.forecast import create_forecast_service
            forecast_service = create_forecast_service()
        
        # Generate forecast
        forecast = forecast_service.generate_forecast(
            location_id=str(location["_id"]),
            location_name=location.get("name", "Unknown"),
            lat=location.get("latitude", 0.0),
            lon=location.get("longitude", 0.0)
        )
        
        return forecast
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Forecast generation failed", location_id=location_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {str(e)}")


@router.get("/timeline/{location_id}", summary="Get hourly risk timeline")
async def get_forecast_timeline(
    request: Request,
    location_id: str,
    hours: int = Query(72, ge=1, le=168, description="Number of hours to forecast (max 168 = 1 week)"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Get hourly flood risk timeline for the next N hours.
    
    Returns interpolated hourly risk scores based on LSTM predictions.
    """
    try:
        from utils.db import get_db
        from bson import ObjectId
        
        db = get_db(request.app)
        
        # Get location details
        try:
            location = await db["locations"].find_one({"_id": ObjectId(location_id)})
        except:
            raise HTTPException(status_code=400, detail="Invalid location ID format")
        
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
        
        # Get forecast service
        forecast_service = getattr(request.app.state, 'forecast_service', None)
        if forecast_service is None:
            from services.forecast import create_forecast_service
            forecast_service = create_forecast_service()
        
        # Generate timeline
        timeline = forecast_service.get_forecast_timeline(
            location_id=str(location["_id"]),
            location_name=location.get("name", "Unknown"),
            lat=location.get("latitude", 0.0),
            lon=location.get("longitude", 0.0),
            hours=hours
        )
        
        return timeline
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Timeline generation failed", location_id=location_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate timeline: {str(e)}")


@router.post("/batch", summary="Generate forecasts for multiple locations")
async def generate_batch_forecasts(
    request: Request,
    payload: BatchForecastRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Generate forecasts for multiple locations in a single request.
    
    Useful for dashboard views or bulk processing.
    """
    try:
        from utils.db import get_db
        from bson import ObjectId
        
        db = get_db(request.app)
        
        # Validate location IDs
        if not payload.location_ids:
            raise HTTPException(status_code=400, detail="No location IDs provided")
        
        if len(payload.location_ids) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 locations allowed per batch request")
        
        # Get locations from database
        locations = []
        for loc_id in payload.location_ids:
            try:
                location = await db["locations"].find_one({"_id": ObjectId(loc_id)})
                if location:
                    locations.append({
                        "id": str(location["_id"]),
                        "name": location.get("name", "Unknown"),
                        "lat": location.get("latitude", 0.0),
                        "lon": location.get("longitude", 0.0)
                    })
            except:
                logger.warning(f"Invalid location ID: {loc_id}")
                continue
        
        if not locations:
            raise HTTPException(status_code=404, detail="No valid locations found")
        
        # Get forecast service
        forecast_service = getattr(request.app.state, 'forecast_service', None)
        if forecast_service is None:
            from services.forecast import create_forecast_service
            forecast_service = create_forecast_service()
        
        # Generate batch forecasts
        forecasts = forecast_service.generate_batch_forecasts(locations)
        
        return {
            "success": True,
            "count": len(forecasts),
            "forecasts": forecasts,
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch forecast failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate batch forecasts: {str(e)}")


@router.get("/early-warnings", summary="Get locations with early warnings")
async def get_early_warnings(
    request: Request,
    min_risk: float = Query(70.0, ge=0.0, le=100.0, description="Minimum risk threshold for warnings"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Get all locations where forecasts indicate high risk or rapid risk increase.
    
    Useful for proactive alert generation and emergency planning.
    """
    try:
        from utils.db import get_db
        
        db = get_db(request.app)
        
        # Get all active locations
        locations = await db["locations"].find(
            {"is_active": True},
            {"_id": 1, "name": 1, "latitude": 1, "longitude": 1}
        ).to_list(length=100)
        
        if not locations:
            return {
                "success": True,
                "count": 0,
                "warnings": [],
                "generated_at": datetime.now().isoformat()
            }
        
        # Get forecast service
        forecast_service = getattr(request.app.state, 'forecast_service', None)
        if forecast_service is None:
            from services.forecast import create_forecast_service
            forecast_service = create_forecast_service()
        
        # Generate forecasts and filter for warnings
        warnings = []
        for location in locations:
            try:
                forecast = forecast_service.generate_forecast(
                    location_id=str(location["_id"]),
                    location_name=location.get("name", "Unknown"),
                    lat=location.get("latitude", 0.0),
                    lon=location.get("longitude", 0.0)
                )
                
                # Check if early warning is flagged or any forecast exceeds threshold
                has_warning = forecast.get("early_warning", False)
                high_risk_forecast = any(
                    f["predicted_risk"] >= min_risk 
                    for f in forecast.get("forecasts", [])
                )
                
                if has_warning or high_risk_forecast:
                    warnings.append(forecast)
                    
            except Exception as e:
                logger.warning(f"Failed to generate forecast for {location.get('name')}: {e}")
                continue
        
        return {
            "success": True,
            "count": len(warnings),
            "warnings": warnings,
            "threshold": min_risk,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Early warnings fetch failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch early warnings: {str(e)}")
