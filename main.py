# main.py - Crisis Connect API
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import time
import structlog
import asyncio
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from config import settings
from utils.db import init_mongo, close_mongo, get_db, ensure_indexes
from middleware import setup_logging_middleware
from services.weather_service import WeatherService
from services.alert_service import AlertService
from services.location_service import LocationService

# Phase 3 imports
from services.alert_formatter import create_alert_formatter
from services.action_generator import create_action_generator
from services.resource_calculator import create_resource_calculator
from services.alert_distributor import create_alert_distributor
from services.emergency_system_integrator import create_emergency_integrator
from services.feedback_system import create_feedback_system

# Import Routers
from routers import weather, alerts, historical, locations, system, forecast, notifications
from services.scheduler import scheduler_service

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
    logger.info("Starting Crisis Connect API", version=settings.api_version)
    app.state.start_time = time.time()
    app.state.executor = ThreadPoolExecutor(max_workers=2)

    # Initialize MongoDB
    try:
        await init_mongo(app)
        db = get_db(app)
        await ensure_indexes(db)
        logger.info("MongoDB initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize MongoDB", error=str(e))

    # Initialize Services
    try:
        app.state.weather_service = WeatherService(app)
        app.state.alert_service = AlertService(app)
        app.state.location_service = LocationService(db)
        logger.info("Core services initialized successfully")
    except Exception as e:
        logger.error("Service initialization failed", error=str(e))
        raise RuntimeError("Service initialization failed")
    
    # Initialize Phase 3 Services
    try:
        app.state.alert_formatter = create_alert_formatter()
        app.state.action_generator = create_action_generator()
        app.state.resource_calculator = create_resource_calculator()
        app.state.alert_distributor = create_alert_distributor()
        app.state.emergency_integrator = create_emergency_integrator()
        app.state.feedback_system = create_feedback_system()
        logger.info("Phase 3 services initialized successfully")
    except Exception as e:
        logger.warning("Phase 3 service initialization failed (non-critical)", error=str(e))

    # Initialize default locations if needed
    try:
        init_result = await app.state.location_service.initialize_default_locations()
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
        if hasattr(app.state, 'redis') and app.state.redis:
            await app.state.redis.close()
            logger.info("Redis connection closed")
    except Exception as e:
        logger.error("Error closing Redis", error=str(e))

    try:
        app.state.executor.shutdown(wait=True)
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

# --- Include Routers ---
# --- Include Routers ---
app.include_router(weather.router)
app.include_router(alerts.router)
app.include_router(historical.router)
app.include_router(locations.router)
app.include_router(system.router)
app.include_router(forecast.router)
app.include_router(notifications.router)

# --- Exception Handlers ---
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

@app.get("/", summary="API Information", tags=["General"])
async def root():
    """Get basic API information"""
    return {
        "message": "Crisis Connect - AI-Powered Flood Prediction System",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
        "api": {
            "weather": "/api/v1/weather",
            "alerts": "/api/v1/alerts",
            "locations": "/api/v1/locations",
            "historical": "/api/v1/historical",
            "system": "/api/v1/system",
            "forecast": "/api/v1/forecast",
            "notifications": "/api/v1/notifications"
        },
        "phase3_endpoints": {
            "actionable_alerts": "/api/v1/alerts/generate-actionable",
            "action_plans": "/api/v1/alerts/actions/generate",
            "resources": "/api/v1/alerts/resources/calculate",
            "distribution": "/api/v1/alerts/distribute",
            "outcomes": "/api/v1/alerts/outcomes",
            "system_metrics": "/api/v1/alerts/system-metrics",
            "feedback_report": "/api/v1/alerts/feedback-report",
            "channels": "/api/v1/alerts/channels",
            "integrations": "/api/v1/alerts/integrations/status"
        }
    }


@app.get("/api/v1", summary="API v1 Overview", tags=["General"])
async def api_v1_overview():
    """Get complete API v1 endpoint overview"""
    return {
        "version": "v1",
        "description": "Crisis Connect API - AI-Powered Disaster Prediction and Emergency Response",
        "endpoints": {
            "weather": {
                "prefix": "/api/v1/weather",
                "description": "Real-time weather data and monitoring",
                "endpoints": [
                    {"method": "GET", "path": "/current/{location}", "description": "Get current weather"},
                    {"method": "GET", "path": "/forecast/{location}", "description": "Get weather forecast"}
                ]
            },
            "alerts": {
                "prefix": "/api/v1/alerts",
                "description": "Alert management and Phase 3 action-oriented alerts",
                "endpoints": [
                    {"method": "GET", "path": "/history", "description": "Get alert history"},
                    {"method": "GET", "path": "/statistics", "description": "Get alert statistics"},
                    {"method": "POST", "path": "/generate", "description": "Generate alerts from predictions"},
                    {"method": "POST", "path": "/generate-actionable", "description": "Generate full actionable alert with action plan"},
                    {"method": "POST", "path": "/distribute", "description": "Distribute alert through channels"},
                    {"method": "POST", "path": "/outcomes", "description": "Record prediction outcomes"},
                    {"method": "GET", "path": "/system-metrics", "description": "Get system performance metrics"},
                    {"method": "GET", "path": "/feedback-report", "description": "Get feedback report"},
                    {"method": "POST", "path": "/actions/generate", "description": "Generate action plan"},
                    {"method": "POST", "path": "/resources/calculate", "description": "Calculate resource requirements"},
                    {"method": "GET", "path": "/channels", "description": "Get distribution channels"},
                    {"method": "GET", "path": "/integrations/status", "description": "Check integration status"}
                ]
            },
            "forecast": {
                "prefix": "/api/v1/forecast",
                "description": "Multi-horizon flood risk forecasting",
                "endpoints": [
                    {"method": "GET", "path": "/{location_id}", "description": "Get forecast for location"},
                    {"method": "GET", "path": "/timeline/{location_id}", "description": "Get hourly risk timeline"},
                    {"method": "POST", "path": "/batch", "description": "Batch forecast generation"},
                    {"method": "GET", "path": "/early-warnings", "description": "Get early warnings"}
                ]
            },
            "locations": {
                "prefix": "/api/v1/locations",
                "description": "Location management",
                "endpoints": [
                    {"method": "GET", "path": "/", "description": "List locations"},
                    {"method": "POST", "path": "/", "description": "Create location"},
                    {"method": "GET", "path": "/{location_id}", "description": "Get location details"}
                ]
            },
            "historical": {
                "prefix": "/api/v1/historical",
                "description": "Historical disaster data",
                "endpoints": [
                    {"method": "GET", "path": "/events", "description": "Get historical events"},
                    {"method": "GET", "path": "/statistics", "description": "Get historical statistics"}
                ]
            },
            "notifications": {
                "prefix": "/api/v1/notifications",
                "description": "Notification configuration",
                "endpoints": [
                    {"method": "POST", "path": "/telegram/config", "description": "Configure Telegram bot"},
                    {"method": "POST", "path": "/telegram/test", "description": "Send test alert"}
                ]
            },
            "system": {
                "prefix": "/",
                "description": "System monitoring and testing",
                "endpoints": [
                    {"method": "GET", "path": "/health", "description": "Health check"},
                    {"method": "GET", "path": "/metrics", "description": "Application metrics"},
                    {"method": "POST", "path": "/api/v1/test/validate-accuracy", "description": "Run accuracy validation"},
                    {"method": "POST", "path": "/api/v1/test/simulate", "description": "Simulate disaster scenario"}
                ]
            }
        }
    }
