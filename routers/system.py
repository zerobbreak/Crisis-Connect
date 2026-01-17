from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
import structlog
from datetime import datetime, timedelta
import time
import asyncio

from utils.dependencies import verify_api_key, rate_limit, get_redis
from utils.db import get_db
from config import settings
from services.predict import DISTRICT_COORDS, calculate_household_resources, SCENARIOS
from models.model import SimulateRequest

logger = structlog.get_logger(__name__)

router = APIRouter(
    tags=["System & Monitoring"]
)

@router.get("/health", summary="Service health check", tags=["Monitoring"])
async def health_check(request: Request):
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.api_version,
        "services": {}
    }
    
    overall_healthy = True
    
    # Check MongoDB
    try:
        db = get_db(request.app)
        await db.command("ping")
        health_status["services"]["mongodb"] = {"status": "healthy", "response_time_ms": 0}
    except Exception as e:
        health_status["services"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
        overall_healthy = False
    
    # Check Redis
    try:
        redis_conn = await get_redis()
        if redis_conn:
            start_time = time.time()
            await redis_conn.ping()
            response_time = (time.time() - start_time) * 1000
            health_status["services"]["redis"] = {"status": "healthy", "response_time_ms": round(response_time, 2)}
        else:
            health_status["services"]["redis"] = {"status": "disabled"}
    except Exception as e:
        health_status["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
    
    # Check ML Model
    model = getattr(request.app.state, "model", None)
    if model is None:
        health_status["services"]["ml_model"] = {"status": "unhealthy", "error": "Model not loaded"}
        overall_healthy = False
    else:
        health_status["services"]["ml_model"] = {"status": "healthy"}
    
    # Check External APIs (Open-Meteo)
    try:
        import requests
        start_time = time.time()
        response = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": -29.8587, "longitude": 31.0218}, timeout=5)
        response_time = (time.time() - start_time) * 1000
        if response.status_code == 200:
            health_status["services"]["open_meteo"] = {"status": "healthy", "response_time_ms": round(response_time, 2)}
        else:
            health_status["services"]["open_meteo"] = {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
            overall_healthy = False
    except Exception as e:
        health_status["services"]["open_meteo"] = {"status": "unhealthy", "error": str(e)}
        overall_healthy = False
    
    if not overall_healthy:
        health_status["status"] = "unhealthy"
        return JSONResponse(status_code=503, content=health_status)
    
    return health_status

@router.get("/metrics", summary="Application metrics", tags=["Monitoring"])
async def get_metrics(request: Request):
    """Get application metrics for monitoring"""
    try:
        db = get_db(request.app)
        
        # Count documents in collections
        collections_stats = {}
        for collection_name in ["weather_data", "predictions", "alerts", "historical_events"]:
            try:
                count = await db[collection_name].count_documents({})
                collections_stats[collection_name] = count
            except Exception:
                collections_stats[collection_name] = 0
        
        # Recent activity (last 24 hours)
        recent_threshold = datetime.now() - timedelta(hours=24)
        recent_weather = await db["weather_data"].count_documents({"timestamp": {"$gte": recent_threshold.isoformat()}})
        recent_predictions = await db["predictions"].count_documents({"timestamp": {"$gte": recent_threshold.isoformat()}})
        recent_alerts = await db["alerts"].count_documents({"timestamp": {"$gte": recent_threshold.isoformat()}})
        
        model = getattr(request.app.state, "model", None)
        redis_client = await get_redis()

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "collections": collections_stats,
            "recent_activity_24h": {
                "weather_data": recent_weather,
                "predictions": recent_predictions,
                "alerts": recent_alerts
            },
            "system": {
                "model_loaded": model is not None,
                "redis_available": redis_client is not None,
                "uptime_seconds": time.time() - request.app.state.start_time if hasattr(request.app.state, 'start_time') else 0
            }
        }
        
        return metrics
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

# --- Testing & Validation ---
@router.post("/api/v1/test/validate-accuracy", summary="Run comprehensive prediction accuracy testing", tags=["Testing"])
async def validate_prediction_accuracy(
    test_days: int = Query(30, ge=7, le=365, description="Number of days to test backward from today"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Run comprehensive accuracy testing against historical data."""
    try:
        logger.info("Starting prediction accuracy validation", test_days=test_days)

        # Import the testing function
        from services.predict import run_prediction_accuracy_test

        # Run the comprehensive test
        results = run_prediction_accuracy_test()

        if 'error' in results:
            raise HTTPException(status_code=500, detail=f"Validation failed: {results['error']}")

        # Return formatted results
        return {
            "success": True,
            "message": "Prediction accuracy validation completed",
            "test_period_days": test_days,
            "validation_results": results['validation_results'],
            "stress_test_results": results['stress_test_results'],
            "edge_case_results": results['edge_case_results'],
            "overall_assessment": results['overall_assessment'],
            "report_saved_to": "prediction_accuracy_report.json",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error("Prediction validation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/api/v1/test/simulate", summary="Simulate a disaster scenario for testing", tags=["Testing"])
async def simulate_disaster_scenario(request: Request, req: SimulateRequest):
    db = get_db(request.app)
    location = req.location.strip()
    scenario = req.scenario
    household_size = req.household_size

    # Get coordinates: from input or fallback to known districts
    lat, lon = None, None
    if req.lat and req.lon:
        lat, lon = float(req.lat), float(req.lon)
    else:
        # Try to match to known district
        matched = None
        for k, (la, lo) in DISTRICT_COORDS.items():
            if location.lower() in k.lower():
                matched = (la, lo)
                break
        if matched:
            lat, lon = matched
        else:
            # Try geocoding
            try:
                loop = asyncio.get_event_loop()
                from geopy.geocoders import Nominatim
                from geopy.extra.rate_limiter import RateLimiter
                _geolocator = Nominatim(user_agent="crisis-connect")
                _geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1)
                
                executor = getattr(request.app.state, "executor", None)
                geo = await loop.run_in_executor(executor, _geocode, location)
                if geo:
                    lat, lon = geo.latitude, geo.longitude
                else:
                    raise HTTPException(status_code=404, detail="Location not found and no coordinates provided")
            except Exception as e:
                logger.warning(f"Geocoding failed: {e}")
                raise HTTPException(status_code=404, detail="Could not resolve location")

    # Build simulated weather
    template = SCENARIOS[scenario]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    simulated_features = {
        "location": location,
        "lat": lat,
        "lon": lon,
        "temp_c": template["temp_c"],
        "humidity": template["humidity"],
        "wind_kph": template["wind_kph"],
        "pressure_mb": template["pressure_mb"],
        "precip_mm": template["precip_mm"],
        "cloud": template["cloud"],
        "wave_height": template["wave_height"],
        "timestamp": now,
        "is_severe": 1,
        "anomaly_score": 95.0,
        "model_risk_score": 88.0,
        "composite_risk_score": 92.0,
        "risk_category": "High",
        "scenario": scenario,
        "description": template["description"]
    }

    # Add household resources
    resources = calculate_household_resources("High", household_size=household_size)
    simulated_features["household_resources"] = resources

    # Save to weather_data and predictions
    from pymongo import UpdateOne
    ops = [
        UpdateOne(
            {"location": location, "scenario": scenario, "timestamp": now},
            {"$set": simulated_features},
            upsert=True
        )
    ]
    await db["simulated_events"].bulk_write(ops)
    await db["weather_data"].bulk_write(ops)
    await db["predictions"].bulk_write(ops)

    logger.info(f"🔥 Simulated {scenario} in {location} (lat={lat}, lon={lon})")

    return {
        "message": f"✅ Simulated {scenario.upper()} in {location}",
        "location": location,
        "coordinates": {"lat": lat, "lon": lon},
        "scenario": scenario,
        "risk_score": 92.0,
        "composite_risk_score": 92.0,
        "household_resources": resources,
        "description": template["description"],
        "timestamp": now
    }
