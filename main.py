# main.py - Crisis Connect API
from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks, Depends, status
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.requests import Request as FastAPIRequest
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, ValidationError
import os
import logging
import structlog
import asyncio
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
import time
from functools import wraps
import hashlib
import math
import requests
import pymongo
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Import services and utilities
from config import settings
from services.weather_service import WeatherService
from services.alert_service import AlertService
from services.location_service import LocationService
from services.predict import DISTRICT_COORDS, collect_all_data, generate_risk_scores, calculate_household_resources
from services.alert_generate import generate_alerts_from_db
from models.model import (
    AlertModel, SimulateRequest, LocationRequest, WeatherEntry,
    LocationCreate, LocationUpdate, Location, LocationPreset, LocationSearch, GeocodeRequest
)
from datetime import datetime, timedelta
from utils.db import init_mongo, close_mongo, get_db, ensure_indexes
from middleware import setup_logging_middleware

# --- Structured Logging Setup ---
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    global model, redis_client, weather_service, alert_service, location_service

    logger.info("Starting Crisis Connect API", version=settings.api_version)
    app.state.start_time = time.time()

    # Initialize MongoDB
    try:
        await init_mongo(app)
        db = get_db(app)
        await ensure_indexes(db)
        logger.info("MongoDB initialized successfully")
    except Exception as e:
        logger.error("MongoDB initialization failed", error=str(e))
        raise RuntimeError("MongoDB initialization failed")

    # Initialize Redis
    try:
        redis_client = redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
        await redis_client.ping()
        logger.info("Redis initialized successfully")
    except Exception as e:
        logger.warning("Redis connection failed, caching disabled", error=str(e))
        redis_client = None

    # Load ML model
    try:
        model = joblib.load(settings.model_path)
        logger.info("ML model loaded successfully")
    except Exception as e:
        logger.error("Model loading failed", error=str(e))
        model = None

    # Initialize Services
    try:
        weather_service = WeatherService(app)
        alert_service = AlertService(app)
        location_service = LocationService(db)
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error("Service initialization failed", error=str(e))
        raise RuntimeError("Service initialization failed")

    # Initialize default locations if needed
    try:
        init_result = await location_service.initialize_default_locations()
        logger.info("Location initialization", result=init_result)
    except Exception as e:
        logger.warning("Location initialization failed", error=str(e))

    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Shutting down Crisis Connect API")

    try:
        await close_mongo(app)
        logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error("Error closing MongoDB", error=str(e))

    try:
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")
    except Exception as e:
        logger.error("Error closing Redis", error=str(e))

    try:
        executor.shutdown(wait=True)
        logger.info("Thread pool executor shutdown")
    except Exception as e:
        logger.error("Error shutting down executor", error=str(e))

    logger.info("Application shutdown completed")

# --- App Setup ---
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup logging middleware
app = setup_logging_middleware(app)

# --- Security & Middleware Setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add trusted host middleware for production
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=settings.trusted_hosts
    )

# --- Security Dependencies ---
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for protected endpoints"""
    if not settings.api_key:
        return True  # No API key required in development
    
    if not credentials or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- Redis Cache Setup ---
redis_client = None

async def get_redis():
    """Get Redis client"""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url(settings.redis_url)
            await redis_client.ping()
        except Exception as e:
            logger.warning("Redis connection failed, caching disabled", error=str(e))
            redis_client = None
    return redis_client

# --- Rate Limiting ---
async def rate_limit(request: FastAPIRequest):
    """Rate limiting based on client IP"""
    if not settings.debug:
        redis_conn = await get_redis()
        if redis_conn:
            client_ip = request.client.host
            key = f"rate_limit:{client_ip}"
            
            current_requests = await redis_conn.get(key)
            if current_requests is None:
                await redis_conn.setex(key, settings.rate_limit_window, 1)
            elif int(current_requests) >= settings.rate_limit_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            else:
                await redis_conn.incr(key)
    return True

# --- Caching Decorator ---
def cache_response(expire_seconds: int = 300):
    """Cache response decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            redis_conn = await get_redis()
            if not redis_conn:
                return await func(*args, **kwargs)
            
            # Create cache key from function name and arguments
            cache_key = f"cache:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get cached result
            cached = await redis_conn.get(cache_key)
            if cached:
                return JSONResponse(content=eval(cached.decode()))
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await redis_conn.setex(cache_key, expire_seconds, str(result))
            return result
        return wrapper
    return decorator

# --- Global Variables ---
model = None  # Will be loaded on startup
executor = ThreadPoolExecutor(max_workers=2)

# --- Geocoder ---
_geolocator = Nominatim(user_agent="crisis-connect")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1)

# --- File Paths ---
HISTORICAL_XLSX = "data_disaster.xlsx"
WEATHER_CSV = "latest_data.csv"
ALERTS_CSV = "alerts.csv"

# --- Helper Functions ---
def serialize_doc(doc):
    """Convert MongoDB doc for JSON serialization."""
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

def _json_safe(value):
    if isinstance(value, (np.floating, float)):
        try:
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except Exception:
            return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

def _df_to_json_records(df: pd.DataFrame) -> List[dict]:
    df = df.replace([float('inf'), float('-inf')], pd.NA)
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    return [{k: _json_safe(v) for k, v in r.items()} for r in records]

def _sanitize_records(records: List[dict]) -> List[dict]:
    return [{k: _json_safe(v) for k, v in r.items()} for r in records]

def _strip_mongo_ids(records: List[dict]) -> List[dict]:
    for r in records:
        r.pop("_id", None)
    return records

# --- Exception Handlers ---
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error("Runtime error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=503, 
        content={
            "error": "Service Unavailable",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning("Validation error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning("HTTP exception", status=exc.status_code, detail=exc.detail, path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", error=str(exc), path=request.url.path, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred" if not settings.debug else str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# --- Health Check & Monitoring ---
@app.get("/health", summary="Service health check", tags=["Monitoring"])
async def health_check():
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
        db = get_db(app)
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

@app.get("/metrics", summary="Application metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics for monitoring"""
    try:
        db = get_db(app)
        
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
                "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
            }
        }
        
        return metrics
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

# --- API Endpoints ---

# API Information
@app.get("/", summary="API Information", tags=["General"])
async def root():
    """Get basic API information"""
    return {
        "message": "Welcome to Crisis Connect API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "api": {
            "v1": "/api/v1/",
            "data_collection": "/api/v1/weather",
            "risk_assessment": "/api/v1/risk",
            "alerts": "/api/v1/alerts",
            "historical": "/api/v1/historical",
            "resources": "/api/v1/resources"
        }
    }


# --- Weather Data Collection (API v1) ---
@app.get("/api/v1/weather/collect", summary="Collect weather data for all active locations", tags=["Data Collection"])
async def collect_weather_data(
    location_ids: Optional[List[str]] = Query(None, description="Specific location IDs to collect data for"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Collect real-time weather data for active locations or specified location IDs"""
    try:
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

@app.post("/api/v1/weather/collect", summary="Collect weather data for custom locations", tags=["Data Collection"])
async def collect_weather_data_custom(
    payload: CollectRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Collect weather data for custom locations or location IDs"""
    try:
        # Validate input - at least one of locations or location_ids must be provided
        if not payload.locations and not payload.location_ids:
            raise HTTPException(status_code=400, detail="Either locations or location_ids must be provided")

        # Validate locations if provided
        validated_locations = None
        if payload.locations:
            if len(payload.locations) > 100:
                raise HTTPException(status_code=400, detail="Too many locations. Maximum 100 allowed.")

            # Validate each location
            validated_locations = []
            for i, loc in enumerate(payload.locations):
                if not isinstance(loc, dict):
                    logger.warning("Invalid location format", index=i, location=loc)
                    continue

                if "lat" not in loc or "lon" not in loc:
                    logger.warning("Missing coordinates", index=i, location=loc)
                    continue

                try:
                    lat = float(loc["lat"])
                    lon = float(loc["lon"])
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        logger.warning("Invalid coordinates", index=i, lat=lat, lon=lon)
                        continue

                    validated_locations.append({
                        "name": loc.get("name", f"Custom_{i}"),
                        "lat": lat,
                        "lon": lon
                    })
                except (ValueError, TypeError) as e:
                    logger.warning("Invalid coordinate values", index=i, error=str(e))
                    continue

            if not validated_locations and not payload.location_ids:
                raise HTTPException(status_code=400, detail="No valid locations provided")

        # Use weather service to collect data
        result = await weather_service.collect_weather_data(
            locations=validated_locations,
            location_ids=payload.location_ids
        )
        
        if result["success"]:
            # Get some sample records for response
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
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to collect custom weather data: {str(e)}"
        )

# --- Risk Assessment (API v1) ---
@app.get("/api/v1/risk/assess", summary="Get latest risk scores for all locations", tags=["Risk Assessment"])
@cache_response(expire_seconds=300)  # Cache for 5 minutes
async def assess_risk_all():
    """Get the latest risk assessment data for all locations"""
    db = get_db(app)
    
    try:
        logger.info("Fetching risk assessment data")
        
        # Get latest predictions with comprehensive data
        predictions = await db["predictions"].find(
            {"composite_risk_score": {"$exists": True, "$ne": None}},
            {
                "_id": 0,
                "location": 1,
                "lat": 1,
                "lon": 1,
                "composite_risk_score": 1,
                "risk_category": 1,
                "model_risk_score": 1,
                "anomaly_score": 1,
                "timestamp": 1,
                "precip_mm": 1,
                "wind_kph": 1,
                "wave_height": 1,
                "household_resources": 1,
                "scenario": 1
            }
        ).sort("timestamp", -1).to_list(length=10000)

        if not predictions:
            logger.warning("No predictions found in database")
            return []

        # Group by location to get latest for each
        latest_predictions = {}
        for pred in predictions:
            location = pred.get("location")
            if location and location not in latest_predictions:
                latest_predictions[location] = pred

        result = list(latest_predictions.values())
        logger.info("Risk assessment data retrieved", 
                   total_predictions=len(predictions),
                   unique_locations=len(result))
        
        return result

    except Exception as e:
        logger.error("Failed to fetch risk assessment data", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Failed to retrieve risk assessments"
        )
    
# --- Risk Prediction (API v1) ---

@app.post("/api/v1/risk/predict", summary="Generate risk predictions using ML models", tags=["Risk Assessment"])
async def predict_risk_scores(
    generate_alerts: bool = Query(False, description="Whether to generate alerts for high-risk locations"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Process available weather data and generate risk assessments using ML models"""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")

    try:
        result = await weather_service.process_risk_assessment(model, generate_alerts)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Risk assessment processing failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process risk assessment: {str(e)}"
        )


# --- Alerts Management (API v1) ---

@app.get("/api/v1/alerts/history", summary="Get alert history", tags=["Alerts"])
@cache_response(expire_seconds=60)  # Cache for 1 minute
async def get_alerts_history(
    location: Optional[str] = Query(None, description="Filter by location name"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    language: Optional[str] = Query(None, description="Filter by language code"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of alerts to return"),
    hours_back: Optional[int] = Query(None, description="Hours of historical data to retrieve")
):
    """Get alert history with filtering options"""
    try:
        result = await alert_service.get_alerts(
            location=location,
            risk_level=risk_level,
            language=language,
            limit=limit,
            hours_back=hours_back
        )
        
        return result
        
    except Exception as e:
        logger.error("Failed to retrieve alerts", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@app.get("/api/v1/alerts/statistics", summary="Get alert statistics", tags=["Alerts"])
@cache_response(expire_seconds=300)  # Cache for 5 minutes
async def get_alerts_statistics(
    days_back: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get alert statistics for monitoring and analytics"""
    try:
        result = await alert_service.get_alert_statistics(days_back)
        return result
        
    except Exception as e:
        logger.error("Failed to get alert statistics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get alert statistics")

@app.post("/api/v1/alerts/generate", summary="Generate alerts from high-risk predictions", tags=["Alerts"])
async def generate_alerts_from_predictions(
    risk_threshold: float = Query(70.0, ge=0, le=100, description="Minimum risk score to generate alerts for"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of predictions to process"),
    include_translations: bool = Query(True, description="Include translated messages using Gemini AI"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Generate alerts from high-risk predictions using ML model data.

    This endpoint:
    - Scans recent predictions for high-risk locations
    - Generates alert messages with household resource requirements
    - Uses Gemini AI for translation to local languages (isiZulu, isiXhosa)
    - Stores alerts in the database for notification systems
    """
    try:
        logger.info("Generating alerts from predictions",
                   risk_threshold=risk_threshold,
                   limit=limit,
                   include_translations=include_translations)

        # Generate alerts using the alert service
        alerts = await alert_service.generate_alerts_from_predictions(limit, risk_threshold)

        if not alerts:
            return {
            "success": True,
                "message": "No high-risk predictions found requiring alerts",
                "alerts_generated": 0,
                "risk_threshold_used": risk_threshold,
                "timestamp": datetime.now().isoformat()
            }

        # Store alerts in database
        db = get_db(app)
        inserted_alerts = []
        for alert in alerts:
            try:
                # Check for duplicates
                existing = await db["alerts"].find_one({
                    "location": alert["location"],
                    "timestamp": alert["timestamp"],
                    "risk_level": alert["risk_level"]
                })

                if not existing:
                    result = await db["alerts"].insert_one(alert)
                    alert["_id"] = str(result.inserted_id)
                    inserted_alerts.append(alert)
                    logger.info("Alert stored in database", location=alert["location"])
                else:
                    logger.warning("Duplicate alert skipped", location=alert["location"])

            except Exception as e:
                logger.error("Failed to store alert", location=alert.get("location"), error=str(e))
                continue

        return {
            "success": True,
            "message": f"Generated {len(inserted_alerts)} new alerts from {len(alerts)} predictions",
            "alerts_generated": len(inserted_alerts),
            "predictions_processed": len(alerts),
            "risk_threshold_used": risk_threshold,
            "translations_included": include_translations,
            "alerts": inserted_alerts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to generate alerts", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate alerts: {str(e)}")

@app.delete("/api/v1/alerts/cleanup", summary="Clean up old alerts", tags=["Alerts"])
async def cleanup_alerts_history(
    days_to_keep: int = Query(90, ge=30, le=365, description="Number of days of alerts to retain"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Delete old alerts to maintain database performance"""
    try:
        result = await alert_service.delete_old_alerts(days_to_keep)
        return result
        
    except Exception as e:
        logger.error("Failed to cleanup old alerts", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cleanup old alerts")

# --- Historical Data (API v1) ---
@app.get("/api/v1/historical/events", response_model=List[dict])
async def get_historical_events():
    """Get all historical disaster events"""
    db = get_db(app)
    docs = await db["historical_events"].find().to_list(length=100000)
    if not docs:
        try:
            df = pd.read_excel(HISTORICAL_XLSX)
            df.columns = df.columns.str.strip().str.lower()
            df.rename(columns={"location": "location"}, inplace=True)
            records = df.to_dict(orient="records")
            if records:
                await db["historical_events"].insert_many(records)
                summary = df.groupby("location")["severity"].value_counts().unstack(fill_value=0).reset_index().to_dict(orient="records")
                await db["historical_summary"].delete_many({})
                await db["historical_summary"].insert_many(summary)
            return records
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise HTTPException(status_code=500, detail="Historical data not available")
    return [_strip_mongo_ids([d])[0] for d in docs]

@app.get("/api/v1/historical/locations")
async def get_historical_locations():
    """Get all location names from historical data"""
    db = get_db(app)
    docs = await db["historical_events"].distinct("location")
    if not docs:
        try:
            df = pd.read_excel(HISTORICAL_XLSX)
            locations = df["location"].dropna().unique().tolist()
            return locations
        except:
            return list(DISTRICT_COORDS.keys())
    return [loc for loc in docs if loc]

@app.get("/api/v1/historical/risk/{location}", description="Assess historical risk profile for a location")
async def assess_historical_risk_by_location(location: str):
    """Get historical disaster risk assessment for a specific location"""
    db = get_db(app)
    docs = await db["historical_events"].find({
        "location": {"$regex": f"^{location}$", "$options": "i"}
    }).to_list(length=10000)

    if not docs:
        try:
            df = pd.read_excel(HISTORICAL_XLSX)
            df.columns = df.columns.str.strip().str.lower()  # Normalize
            required_cols = {'location', 'total_deaths'}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Missing required columns. Found: {list(df.columns)}")

            # Handle missing values
            df['total_deaths'] = pd.to_numeric(df['total_deaths'], errors='coerce').fillna(0)
            df['location'] = df['location'].astype(str).str.strip()

            filtered = df[df['location'].str.contains(location, case=False, na=False)]
            if filtered.empty:
                raise HTTPException(status_code=404, detail="No historical data for this location")

            severity_count = filtered["severity"].value_counts().to_dict() if "severity" in filtered.columns else {"Unknown": len(filtered)}
            total = int(filtered['total_deaths'].sum())

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise HTTPException(status_code=500, detail="Data load failed")
    else:
        df = pd.DataFrame(_strip_mongo_ids(docs))
        severity_count = df["severity"].value_counts().to_dict() if "severity" in df.columns else {"Unknown": len(df)}
        total = len(df)

    return {
        "location": location,
        "total_events": total,
        "risk_profile": severity_count
    }
# --- Resource Calculator (API v1) ---
class ResourceRequest(BaseModel):
    place_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    household_size: int = 4

@app.post("/api/v1/resources/calculate", summary="Calculate household resource needs")
async def calculate_household_resources(request: ResourceRequest):
    db = get_db(app)
    location = request.place_name
    lat = request.lat
    lon = request.lon
    household_size = request.household_size

    if household_size < 1:
        raise HTTPException(status_code=400, detail="Household size must be >= 1")

    # Resolve location to coordinates
    if location and not (lat and lon):
        try:
            loop = asyncio.get_event_loop()
            geocode_result = await loop.run_in_executor(executor, _geocode, location)
            if geocode_result:
                lat, lon = geocode_result.latitude, geocode_result.longitude
            else:
                raise HTTPException(status_code=404, detail="Location not found")
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
            raise HTTPException(status_code=400, detail="Geocoding failed")

    if not (lat and lon):
        raise HTTPException(status_code=400, detail="Valid coordinates required")

    # Find latest prediction
    query = {
        "lat": {"$gte": lat - 0.1, "$lte": lat + 0.1},
        "lon": {"$gte": lon - 0.1, "$lte": lon + 0.1}
    }
    prediction_doc = await db["predictions"].find(query).sort("timestamp", -1).limit(1).to_list(1)
    
    if not prediction_doc:
        # Fallback: collect fresh data
        try:
            df = collect_all_data({location or "custom": (lat, lon)})
            if df.empty:
                resources = calculate_household_resources("Low", household_size)
                return {
                    "location": location or "Custom",
                    "risk_category": "Low",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "household_size": household_size,
                    "resources": resources,
                    "message": "No data; assuming Low risk"
                }
            df = generate_risk_scores(model, df)
            prediction = df.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Fresh collect failed: {e}")
            raise HTTPException(status_code=502, detail="Data collection failed")
    else:
        prediction = _strip_mongo_ids(prediction_doc)[0]

    risk_category = prediction.get("risk_category", "Low")
    try:
        resources = calculate_household_resources(risk_category, household_size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response = {
        "location": prediction.get("location", location or "Custom"),
        "risk_category": risk_category,
        "timestamp": prediction.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "household_size": household_size,
        "resources": resources,
        **{k: v for k, v in prediction.items() if k in ['anomaly_score', 'precip_mm', 'wind_kph', 'wave_height', 'model_risk_score']}
    }
    return response

# --- Legacy Alert Generation (to be removed after migration) ---
@app.post("/alerts/generate")
async def trigger_alert_generation_legacy(limit: int = Query(500)):
    """Legacy endpoint - use /api/v1/alerts/generate instead"""
    db = get_db(app)
    try:
        alerts = await generate_alerts_from_db(db, limit=limit, risk_threshold=70.0)
        return {"generated": len(alerts), "alerts": [_strip_mongo_ids([a])[0] for a in alerts]}
    except Exception as e:
        logger.error(f"Alert generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate alerts")


# Disaster scenario templates
SCENARIOS = {
    "flood": {
        "precip_mm": 60.0,
        "temp_c": 18.0,
        "humidity": 92.0,
        "wind_kph": 30.0,
        "pressure_mb": 1000.0,
        "cloud": 95.0,
        "wave_height": 0.5,
        "description": "Severe inland flooding due to prolonged rainfall"
    },
    "storm": {
        "precip_mm": 40.0,
        "temp_c": 22.0,
        "humidity": 85.0,
        "wind_kph": 80.0,
        "pressure_mb": 995.0,
        "cloud": 100.0,
        "wave_height": 1.2,
        "description": "Intense thunderstorm with strong winds"
    },
    "coastal_cyclone": {
        "precip_mm": 50.0,
        "temp_c": 25.0,
        "humidity": 90.0,
        "wind_kph": 110.0,
        "pressure_mb": 980.0,
        "cloud": 100.0,
        "wave_height": 4.5,
        "description": "Coastal cyclone with storm surge"
    },
    "flash_flood": {
        "precip_mm": 80.0,
        "temp_c": 20.0,
        "humidity": 95.0,
        "wind_kph": 40.0,
        "pressure_mb": 990.0,
        "cloud": 98.0,
        "wave_height": 0.3,
        "description": "Extreme flash flooding in urban/rural areas"
    }
}

# --- Testing & Validation ---
@app.post("/api/v1/test/validate-accuracy", summary="Run comprehensive prediction accuracy testing")
async def validate_prediction_accuracy(
    test_days: int = Query(30, ge=7, le=365, description="Number of days to test backward from today"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Run comprehensive accuracy testing against historical data.

    This endpoint:
    - Tests predictions against known flood events from the past
    - Calculates accuracy, precision, recall, and F1 scores
    - Runs stress tests with extreme weather conditions
    - Tests edge cases and error handling
    - Generates detailed accuracy report

    Returns comprehensive validation results including:
    - Overall accuracy metrics
    - Confusion matrix
    - Risk score distributions
    - Performance assessment and recommendations
    """
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

# --- Testing & Simulation ---
@app.post("/api/v1/test/simulate", summary="Simulate a disaster scenario for testing")
async def simulate_disaster_scenario(request: SimulateRequest):
    db = get_db(app)
    location = request.location.strip()
    scenario = request.scenario
    household_size = request.household_size

    # Get coordinates: from input or fallback to known districts
    lat, lon = None, None
    if request.lat and request.lon:
        lat, lon = float(request.lat), float(request.lon)
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

# --- Location Management (API v1) ---

@app.post("/api/v1/locations", summary="Create new location", tags=["Location Management"])
async def create_location(
    location_data: LocationCreate,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Create a new location with coordinates and metadata"""
    try:
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

@app.get("/api/v1/locations/{location_id}", summary="Get location by ID", tags=["Location Management"])
async def get_location(
    location_id: str,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Get location details by ID"""
    try:
        location = await location_service.get_location(location_id)
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
        return {"success": True, "location": location}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get location failed", location_id=location_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get location: {str(e)}")

@app.put("/api/v1/locations/{location_id}", summary="Update location", tags=["Location Management"])
async def update_location(
    location_id: str,
    update_data: LocationUpdate,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Update location details"""
    try:
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

@app.delete("/api/v1/locations/{location_id}", summary="Delete location", tags=["Location Management"])
async def delete_location(
    location_id: str,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Soft delete a location"""
    try:
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

@app.post("/api/v1/locations/search", summary="Search locations", tags=["Location Management"])
async def search_locations(
    search: LocationSearch,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Search and filter locations with advanced criteria"""
    try:
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

@app.post("/api/v1/locations/geocode", summary="Geocode location name", tags=["Location Management"])
async def geocode_location(
    request: GeocodeRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Find coordinates for a location name using geocoding"""
    try:
        result = await location_service.geocode_location(request)
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=404, detail=result["message"])
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Geocoding failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Geocoding failed: {str(e)}")

@app.post("/api/v1/locations/presets", summary="Create location preset", tags=["Location Management"])
async def create_location_preset(
    preset_data: LocationPreset,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Create a preset collection of locations for easy access"""
    try:
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

@app.get("/api/v1/locations/presets/{preset_id}", summary="Get preset locations", tags=["Location Management"])
async def get_preset_locations(
    preset_id: str,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Get all locations in a preset"""
    try:
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

@app.get("/api/v1/locations/initialize", summary="Initialize default locations", tags=["Location Management"])
async def initialize_locations(
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Initialize database with default South African locations"""
    try:
        result = await location_service.initialize_default_locations()
        return result
    except Exception as e:
        logger.error("Location initialization failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to initialize locations: {str(e)}")

@app.get("/api/v1/locations/coords", summary="Get location coordinates", tags=["Location Management"])
async def get_location_coordinates(
    location_ids: Optional[List[str]] = Query(None, description="Specific location IDs to get coordinates for"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Get coordinates for locations (compatible with prediction system)"""
    try:
        coords = await location_service.get_locations_coords(location_ids)
        return {
            "success": True,
            "coordinates": coords,
            "count": len(coords)
        }
    except Exception as e:
        logger.error("Get coordinates failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get coordinates: {str(e)}")