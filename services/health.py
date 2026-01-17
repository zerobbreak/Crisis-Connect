"""
Service Health Check Module
Provides health check functionality for all services
"""

import structlog
from typing import Dict, Any
from datetime import datetime
import asyncio

logger = structlog.get_logger(__name__)


class ServiceHealth:
    """Health check utilities for services"""
    
    @staticmethod
    async def check_database(db) -> Dict[str, Any]:
        """
        Check database connectivity and performance
        
        Args:
            db: MongoDB database instance
            
        Returns:
            Health check result
        """
        try:
            start = datetime.utcnow()
            await db.command('ping')
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def check_redis(redis) -> Dict[str, Any]:
        """
        Check Redis connectivity
        
        Args:
            redis: Redis client instance
            
        Returns:
            Health check result
        """
        if not redis:
            return {
                "status": "disabled",
                "message": "Redis not configured"
            }
        
        try:
            start = datetime.utcnow()
            await redis.ping()
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def check_weather_service(service) -> Dict[str, Any]:
        """
        Check weather service health
        
        Args:
            service: WeatherService instance
            
        Returns:
            Health check result
        """
        try:
            # Quick validation without actual API call
            if not service.db:
                return {
                    "status": "unhealthy",
                    "error": "Database not connected"
                }
            
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def check_alert_service(service) -> Dict[str, Any]:
        """
        Check alert service health
        
        Args:
            service: AlertService instance
            
        Returns:
            Health check result
        """
        try:
            if not service.db:
                return {
                    "status": "unhealthy",
                    "error": "Database not connected"
                }
            
            # Check if we can query alerts
            start = datetime.utcnow()
            await service.db.alerts.count_documents({}, limit=1)
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def check_all_services(app) -> Dict[str, Any]:
        """
        Check health of all services
        
        Args:
            app: FastAPI app instance
            
        Returns:
            Comprehensive health check results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        # Check database
        if hasattr(app.state, 'db'):
            results["services"]["database"] = await ServiceHealth.check_database(app.state.db)
        
        # Check Redis
        if hasattr(app.state, 'redis'):
            results["services"]["redis"] = await ServiceHealth.check_redis(app.state.redis)
        
        # Check weather service
        if hasattr(app.state, 'weather_service'):
            results["services"]["weather"] = await ServiceHealth.check_weather_service(
                app.state.weather_service
            )
        
        # Check alert service
        if hasattr(app.state, 'alert_service'):
            results["services"]["alerts"] = await ServiceHealth.check_alert_service(
                app.state.alert_service
            )
        
        # Determine overall status
        statuses = [s.get("status") for s in results["services"].values()]
        if all(s == "healthy" for s in statuses if s != "disabled"):
            results["overall_status"] = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            results["overall_status"] = "degraded"
        else:
            results["overall_status"] = "unknown"
        
        return results
