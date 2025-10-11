"""
Alert management service for Crisis Connect API
"""
import asyncio
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import structlog

from utils.db import get_db
from models.model import AlertModel

logger = structlog.get_logger(__name__)


class AlertService:
    """Service for alert management and processing"""
    
    def __init__(self, app):
        self.app = app
        self.db = get_db(app)
    
    async def create_alert(self, alert_data: AlertModel) -> Dict[str, Any]:
        """
        Create a new alert
        
        Args:
            alert_data: Alert model with validated data
            
        Returns:
            Dictionary with creation results
        """
        try:
            logger.info("Creating new alert", location=alert_data.location, risk_level=alert_data.risk_level)
            
            # Convert to dictionary
            alert_dict = alert_data.dict()
            
            # Check for duplicate alerts
            existing = await self.db["alerts"].find_one({
                "location": alert_dict["location"],
                "timestamp": alert_dict["timestamp"],
                "risk_level": alert_dict["risk_level"]
            })
            
            if existing:
                logger.warning("Duplicate alert detected", location=alert_dict["location"])
                raise ValueError("Alert already exists for this location and timestamp")
            
            # Insert alert
            result = await self.db["alerts"].insert_one(alert_dict)
            alert_dict["_id"] = str(result.inserted_id)
            
            logger.info("Alert created successfully", alert_id=result.inserted_id)
            
            return {
                "success": True,
                "message": "Alert created successfully",
                "alert": alert_dict,
                "timestamp": datetime.now().isoformat()
            }
            
        except ValueError as e:
            logger.warning("Alert creation failed - validation error", error=str(e))
            raise
        except Exception as e:
            logger.error("Alert creation failed", error=str(e), exc_info=True)
            raise
    
    async def get_alerts(self,
                        location: Optional[str] = None,
                        risk_level: Optional[str] = None,
                        language: Optional[str] = None,
                        limit: int = 100,
                        hours_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve alerts with filtering options
        
        Args:
            location: Filter by location name
            risk_level: Filter by risk level (LOW, MODERATE, HIGH, SEVERE)
            language: Filter by language code
            limit: Maximum number of alerts to return
            hours_back: Hours of historical data to retrieve
            
        Returns:
            Dictionary with alerts and metadata
        """
        try:
            logger.info("Retrieving alerts", 
                       location=location,
                       risk_level=risk_level,
                       language=language,
                       limit=limit)
            
            # Build query
            query = {}
            
            if location:
                query["location"] = {"$regex": f"^{location}$", "$options": "i"}
            
            if risk_level:
                query["risk_level"] = risk_level.upper()
            
            if language:
                query["language"] = language.lower()
            
            # Add time filter if specified
            if hours_back:
                time_threshold = datetime.now() - timedelta(hours=hours_back)
                query["timestamp"] = {"$gte": time_threshold.strftime("%Y-%m-%d %H:%M:%S")}
            
            # Execute query
            cursor = self.db["alerts"].find(query).sort("timestamp", -1).limit(limit)
            alerts = await cursor.to_list(length=limit)
            
            # Remove MongoDB IDs for JSON serialization
            for alert in alerts:
                alert.pop("_id", None)
            
            logger.info("Alerts retrieved successfully", count=len(alerts))
            
            return {
                "success": True,
                "count": len(alerts),
                "alerts": alerts,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to retrieve alerts", error=str(e), exc_info=True)
            raise
    
    async def generate_alerts_from_predictions(self, limit: int = 1000, risk_threshold: float = 70.0) -> List[Dict[str, Any]]:
        """
        Generate alerts from high-risk predictions

        Args:
            limit: Maximum number of predictions to process
            risk_threshold: Minimum risk score to generate alerts for

        Returns:
            List of generated alerts
        """
        try:
            logger.info("Generating alerts from predictions", limit=limit, risk_threshold=risk_threshold)
            start_time = time.time()

            # Import here to avoid circular imports
            from services.alert_generate import generate_alerts_from_db

            # Generate alerts using existing function with threshold
            alerts = await generate_alerts_from_db(self.db, limit=limit, risk_threshold=risk_threshold)

            duration = time.time() - start_time
            logger.info("Alert generation completed",
                       alerts_generated=len(alerts),
                       risk_threshold=risk_threshold,
                       duration_seconds=round(duration, 2))

            return alerts

        except Exception as e:
            logger.error("Alert generation failed", error=str(e), exc_info=True)
            raise
    
    async def get_alert_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Get alert statistics for monitoring
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with alert statistics
        """
        try:
            logger.info("Calculating alert statistics", days_back=days_back)
            
            # Time range
            start_date = datetime.now() - timedelta(days=days_back)
            
            # Total alerts in period
            total_alerts = await self.db["alerts"].count_documents({
                "timestamp": {"$gte": start_date.strftime("%Y-%m-%d %H:%M:%S")}
            })
            
            # Alerts by risk level
            risk_level_stats = {}
            for level in ["LOW", "MODERATE", "HIGH", "SEVERE"]:
                count = await self.db["alerts"].count_documents({
                    "timestamp": {"$gte": start_date.strftime("%Y-%m-%d %H:%M:%S")},
                    "risk_level": level
                })
                risk_level_stats[level] = count
            
            # Alerts by location
            location_pipeline = [
                {"$match": {"timestamp": {"$gte": start_date.strftime("%Y-%m-%d %H:%M:%S")}}},
                {"$group": {"_id": "$location", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            location_stats = await self.db["alerts"].aggregate(location_pipeline).to_list(length=10)
            
            # Recent activity (last 24 hours)
            recent_threshold = datetime.now() - timedelta(hours=24)
            recent_alerts = await self.db["alerts"].count_documents({
                "timestamp": {"$gte": recent_threshold.strftime("%Y-%m-%d %H:%M:%S")}
            })
            
            stats = {
                "period_days": days_back,
                "total_alerts": total_alerts,
                "recent_alerts_24h": recent_alerts,
                "risk_level_distribution": risk_level_stats,
                "top_locations": location_stats,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Alert statistics calculated", total_alerts=total_alerts)
            
            return stats
            
        except Exception as e:
            logger.error("Failed to calculate alert statistics", error=str(e), exc_info=True)
            raise
    
    async def delete_old_alerts(self, days_to_keep: int = 90) -> Dict[str, Any]:
        """
        Delete old alerts to maintain database performance
        
        Args:
            days_to_keep: Number of days of alerts to retain
            
        Returns:
            Dictionary with deletion results
        """
        try:
            logger.info("Cleaning up old alerts", days_to_keep=days_to_keep)
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old alerts
            result = await self.db["alerts"].delete_many({
                "timestamp": {"$lt": cutoff_date.strftime("%Y-%m-%d %H:%M:%S")}
            })
            
            deleted_count = result.deleted_count
            
            logger.info("Old alerts cleanup completed", deleted_count=deleted_count)
            
            return {
                "success": True,
                "message": f"Deleted {deleted_count} old alerts",
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to delete old alerts", error=str(e), exc_info=True)
            raise
