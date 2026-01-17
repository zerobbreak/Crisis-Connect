from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import Optional
import structlog
from datetime import datetime

from utils.dependencies import verify_api_key, rate_limit, cache_response
from services.alert_service import AlertService
from utils.db import get_db

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/alerts",
    tags=["Alerts"]
)

@router.get("/history", summary="Get alert history")
@cache_response(expire_seconds=60)
async def get_alerts_history(
    request: Request,
    location: Optional[str] = Query(None, description="Filter by location name"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    language: Optional[str] = Query(None, description="Filter by language code"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of alerts to return"),
    hours_back: Optional[int] = Query(None, description="Hours of historical data to retrieve")
):
    """Get alert history with filtering options"""
    try:
        alert_service: AlertService = request.app.state.alert_service
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

@router.get("/statistics", summary="Get alert statistics")
@cache_response(expire_seconds=300)
async def get_alerts_statistics(
    request: Request,
    days_back: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get alert statistics for monitoring and analytics"""
    try:
        alert_service: AlertService = request.app.state.alert_service
        result = await alert_service.get_alert_statistics(days_back)
        return result
    except Exception as e:
        logger.error("Failed to get alert statistics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get alert statistics")

@router.post("/generate", summary="Generate alerts from high-risk predictions")
async def generate_alerts_from_predictions(
    request: Request,
    risk_threshold: float = Query(70.0, ge=0, le=100, description="Minimum risk score to generate alerts for"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of predictions to process"),
    include_translations: bool = Query(True, description="Include translated messages using Gemini AI"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Generate alerts from high-risk predictions using ML model data."""
    try:
        alert_service: AlertService = request.app.state.alert_service
        logger.info("Generating alerts from predictions", risk_threshold=risk_threshold, limit=limit)

        alerts = await alert_service.generate_alerts_from_predictions(limit, risk_threshold)

        if not alerts:
            return {
                "success": True,
                "message": "No high-risk predictions found requiring alerts",
                "alerts_generated": 0,
                "timestamp": datetime.now().isoformat()
            }

        # Store alerts in database
        db = get_db(request.app)
        inserted_alerts = []
        for alert in alerts:
            try:
                existing = await db["alerts"].find_one({
                    "location": alert["location"],
                    "timestamp": alert["timestamp"],
                    "risk_level": alert["risk_level"]
                })

                if not existing:
                    result = await db["alerts"].insert_one(alert)
                    alert["_id"] = str(result.inserted_id)
                    inserted_alerts.append(alert)
            except Exception as e:
                logger.error("Failed to store alert", location=alert.get("location"), error=str(e))
                continue

        return {
            "success": True,
            "message": f"Generated {len(inserted_alerts)} new alerts from {len(alerts)} predictions",
            "alerts_generated": len(inserted_alerts),
            "predictions_processed": len(alerts),
            "alerts": inserted_alerts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to generate alerts", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate alerts: {str(e)}")

@router.delete("/cleanup", summary="Clean up old alerts")
async def cleanup_alerts_history(
    request: Request,
    days_to_keep: int = Query(90, ge=30, le=365, description="Number of days of alerts to retain"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Delete old alerts to maintain database performance"""
    try:
        alert_service: AlertService = request.app.state.alert_service
        result = await alert_service.delete_old_alerts(days_to_keep)
        return result
    except Exception as e:
        logger.error("Failed to cleanup old alerts", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cleanup old alerts")

@router.post("/{alert_id}/feedback", summary="Submit feedback for an alert")
async def submit_alert_feedback(
    request: Request,
    alert_id: str,
    accurate: bool = Query(..., description="Whether the alert was accurate"),
    comments: Optional[str] = Query(None, description="Optional comments"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Submit user feedback for an alert.
    This is critical for the Continuous Learning system to label data.
    """
    try:
        db = get_db(request.app)
        from bson import ObjectId
        
        # Update the alert with feedback
        result = await db["alerts"].update_one(
            {"_id": ObjectId(alert_id)},
            {
                "$set": {
                    "feedback": {
                        "accurate": accurate,
                        "comments": comments,
                        "timestamp": datetime.now().isoformat()
                    },
                    "verified": True
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Alert not found")
            
        return {
            "success": True, 
            "message": "Feedback recorded successfully",
            "alert_id": alert_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to submit feedback", alert_id=alert_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")
