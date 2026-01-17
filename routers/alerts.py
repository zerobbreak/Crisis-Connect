from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import structlog
from datetime import datetime

from utils.dependencies import verify_api_key, rate_limit, cache_response
from services.alert_service import AlertService
from utils.db import get_db

# Phase 3 imports
from services.alert_formatter import AlertFormatter, create_alert_formatter
from services.action_generator import ActionGenerator, create_action_generator
from services.resource_calculator import ResourceCalculator, create_resource_calculator
from services.alert_distributor import AlertDistributor, create_alert_distributor
from services.emergency_system_integrator import EmergencySystemIntegrator, create_emergency_integrator
from services.feedback_system import FeedbackSystem, PredictionOutcome, create_feedback_system

logger = structlog.get_logger(__name__)

# Initialize Phase 3 services
_alert_formatter = create_alert_formatter()
_action_generator = create_action_generator()
_resource_calculator = create_resource_calculator()
_alert_distributor = create_alert_distributor()
_emergency_integrator = create_emergency_integrator()
_feedback_system = create_feedback_system()


# Pydantic models for Phase 3 endpoints
class ActionableAlertRequest(BaseModel):
    """Request model for generating actionable alerts"""
    location: str = Field(..., description="Location name or ID")
    hazard_type: str = Field(default="flood", description="Type of hazard")
    distribute: bool = Field(default=False, description="Whether to distribute the alert")
    integrate: bool = Field(default=False, description="Whether to integrate with emergency systems")
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": "Johannesburg",
                "hazard_type": "flood",
                "distribute": False,
                "integrate": False
            }
        }


class RecipientModel(BaseModel):
    """Recipient for alert distribution"""
    type: str = Field(..., description="Recipient type: residential, authority, critical_facility")
    identifier: str = Field(..., description="Phone number, email, or other identifier")
    channels: Optional[List[str]] = Field(None, description="Specific channels to use")


class DistributeAlertRequest(BaseModel):
    """Request model for distributing an alert"""
    alert: Dict[str, Any] = Field(..., description="Alert to distribute")
    recipients: List[RecipientModel] = Field(..., description="List of recipients")


class OutcomeRequest(BaseModel):
    """Request model for recording prediction outcomes"""
    prediction_id: str
    alert_id: str
    location: str
    hazard_type: str = "flood"
    prediction_timestamp: str
    
    # Prediction details
    predicted_risk_level: str
    predicted_risk_score: float
    predicted_peak_hours: int
    predicted_severity: str
    method_breakdown: Dict[str, float] = Field(default_factory=dict)
    
    # Actual outcome
    actual_disaster_occurred: bool
    actual_peak_time: Optional[str] = None
    actual_severity: Optional[str] = None
    actual_damage_estimate: float = 0.0
    actual_affected_population: int = 0
    actual_casualties: int = 0
    actual_injuries: int = 0
    
    # Action effectiveness
    actions_planned: int = 0
    actions_executed: int = 0
    actions_successful: int = 0
    evacuation_ordered: bool = False
    evacuation_rate: float = 0.0
    shelter_utilization: float = 0.0
    
    # Timing
    lead_time_hours: int = 0
    lead_time_used_hours: int = 0
    
    # Assessment
    false_alarm: bool = False
    missed_event: bool = False
    confidence_rating: float = 0.0
    
    notes: str = ""
    recorded_by: str = ""
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction_id": "PRED-123",
                "alert_id": "CrisisConnect-Johannesburg-20260117-1",
                "location": "Johannesburg",
                "hazard_type": "flood",
                "prediction_timestamp": "2026-01-17T10:00:00Z",
                "predicted_risk_level": "HIGH",
                "predicted_risk_score": 0.78,
                "predicted_peak_hours": 18,
                "predicted_severity": "Severe",
                "actual_disaster_occurred": True,
                "actual_severity": "Severe",
                "actions_planned": 10,
                "actions_executed": 8,
                "actions_successful": 7,
                "evacuation_ordered": True,
                "evacuation_rate": 0.85,
                "lead_time_hours": 18,
                "lead_time_used_hours": 12,
                "false_alarm": False,
                "confidence_rating": 0.8
            }
        }

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


# ============================================================================
# Phase 3: Action-Oriented Alerts & Emergency Integration Endpoints
# ============================================================================

@router.post("/generate-actionable", summary="Generate actionable alert with full action plan")
async def generate_actionable_alert(
    request: Request,
    alert_request: ActionableAlertRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Phase 3 endpoint: Generate a complete actionable alert
    
    Takes a location and hazard type, generates a Phase 2 prediction,
    then transforms it into:
    1. CAP-compliant formatted alert
    2. Specific action plan with stakeholders and timelines
    3. Resource requirements with gap analysis
    4. Optional: Distributes through multiple channels
    5. Optional: Integrates with emergency systems
    """
    try:
        logger.info("Generating actionable alert", 
                   location=alert_request.location, 
                   hazard_type=alert_request.hazard_type)
        
        # Get Phase 2 prediction using orchestrator
        # First, try to get from app state, otherwise create mock prediction
        prediction = await _get_phase2_prediction(request, alert_request.location, alert_request.hazard_type)
        
        # Get location data
        location_data = await _get_location_data(request, alert_request.location)
        
        # Step 1: Format alert (CAP-compliant)
        formatted_alert = _alert_formatter.format_prediction(prediction, location_data)
        
        # Step 2: Generate action plan
        action_plan = _action_generator.generate_actions(prediction, location_data)
        
        # Step 3: Calculate resources
        resource_report = _resource_calculator.calculate_resources(
            prediction, location_data, [a.to_dict() for a in action_plan.actions]
        )
        
        # Prepare response
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "alert": formatted_alert.to_dict(),
            "action_plan": action_plan.to_dict(),
            "resources": resource_report.to_dict()
        }
        
        # Step 4: Optional distribution
        if alert_request.distribute:
            recipients = await _get_alert_recipients(request, alert_request.location, prediction)
            distribution_report = await _alert_distributor.distribute_alert(
                formatted_alert.to_simple_dict(),
                recipients
            )
            response["distribution"] = distribution_report.to_dict()
        
        # Step 5: Optional integration with emergency systems
        if alert_request.integrate:
            integration_report = await _emergency_integrator.integrate_alert(
                formatted_alert.to_dict(),
                [a.to_dict() for a in action_plan.actions],
                resource_report.to_dict()
            )
            response["integration"] = integration_report.to_dict()
        
        logger.info("Actionable alert generated successfully",
                   alert_id=formatted_alert.identifier,
                   actions=len(action_plan.actions),
                   resources=len(resource_report.resources))
        
        return response
        
    except Exception as e:
        logger.error("Failed to generate actionable alert", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate actionable alert: {str(e)}")


@router.post("/distribute", summary="Distribute an alert through multiple channels")
async def distribute_alert(
    request: Request,
    distribute_request: DistributeAlertRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Distribute an alert through multiple channels
    
    Channels include: SMS, Email, Dashboard, WhatsApp, Phone Tree, Sirens
    """
    try:
        logger.info("Distributing alert", 
                   alert_id=distribute_request.alert.get('identifier', 'unknown'),
                   recipients=len(distribute_request.recipients))
        
        # Convert recipients to dict format
        recipients = [r.model_dump() for r in distribute_request.recipients]
        
        # Distribute
        report = await _alert_distributor.distribute_alert(
            distribute_request.alert,
            recipients
        )
        
        return {
            "success": True,
            "distribution": report.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to distribute alert", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to distribute alert: {str(e)}")


@router.post("/outcomes", summary="Record prediction outcome for model improvement")
async def record_prediction_outcome(
    request: Request,
    outcome_data: OutcomeRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Record what actually happened after a prediction
    
    This is critical for the feedback loop:
    - Tracks prediction accuracy
    - Measures action effectiveness
    - Identifies areas for improvement
    - Enables model retraining
    """
    try:
        logger.info("Recording prediction outcome",
                   prediction_id=outcome_data.prediction_id,
                   disaster_occurred=outcome_data.actual_disaster_occurred)
        
        # Convert to PredictionOutcome
        outcome = PredictionOutcome(
            prediction_id=outcome_data.prediction_id,
            alert_id=outcome_data.alert_id,
            location=outcome_data.location,
            hazard_type=outcome_data.hazard_type,
            prediction_timestamp=outcome_data.prediction_timestamp,
            predicted_risk_level=outcome_data.predicted_risk_level,
            predicted_risk_score=outcome_data.predicted_risk_score,
            predicted_peak_hours=outcome_data.predicted_peak_hours,
            predicted_severity=outcome_data.predicted_severity,
            method_breakdown=outcome_data.method_breakdown,
            actual_disaster_occurred=outcome_data.actual_disaster_occurred,
            actual_peak_time=outcome_data.actual_peak_time,
            actual_severity=outcome_data.actual_severity,
            actual_damage_estimate=outcome_data.actual_damage_estimate,
            actual_affected_population=outcome_data.actual_affected_population,
            actual_casualties=outcome_data.actual_casualties,
            actual_injuries=outcome_data.actual_injuries,
            actions_planned=outcome_data.actions_planned,
            actions_executed=outcome_data.actions_executed,
            actions_successful=outcome_data.actions_successful,
            evacuation_ordered=outcome_data.evacuation_ordered,
            evacuation_rate=outcome_data.evacuation_rate,
            shelter_utilization=outcome_data.shelter_utilization,
            lead_time_hours=outcome_data.lead_time_hours,
            lead_time_used_hours=outcome_data.lead_time_used_hours,
            false_alarm=outcome_data.false_alarm,
            missed_event=outcome_data.missed_event,
            confidence_rating=outcome_data.confidence_rating,
            notes=outcome_data.notes,
            recorded_by=outcome_data.recorded_by
        )
        
        # Record outcome
        result = await _feedback_system.record_outcome(outcome)
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to record outcome", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to record outcome: {str(e)}")


@router.get("/system-metrics", summary="Get overall system performance metrics")
@cache_response(expire_seconds=300)
async def get_system_metrics(request: Request):
    """
    Get comprehensive system performance metrics
    
    Returns:
    - Prediction accuracy (precision, recall, F1)
    - False alarm and miss rates
    - Action execution effectiveness
    - Evacuation success rates
    - Lives saved estimates
    - Improvement recommendations
    """
    try:
        metrics = _feedback_system.calculate_system_metrics()
        improvements = _feedback_system.identify_pattern_improvements()
        
        return {
            "success": True,
            "metrics": metrics.to_dict(),
            "recommended_improvements": improvements,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


@router.get("/feedback-report", summary="Get comprehensive feedback report")
async def get_feedback_report(
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Number of days to include in report")
):
    """
    Generate comprehensive feedback report for specified period
    """
    try:
        report = _feedback_system.generate_feedback_report(days)
        
        return {
            "success": True,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to generate feedback report", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate feedback report: {str(e)}")


@router.get("/recent-outcomes", summary="Get recent prediction outcomes")
async def get_recent_outcomes(
    request: Request,
    limit: int = Query(10, ge=1, le=100, description="Number of outcomes to return")
):
    """
    Get most recent prediction outcomes
    """
    try:
        outcomes = _feedback_system.get_recent_outcomes(limit)
        
        return {
            "success": True,
            "count": len(outcomes),
            "outcomes": outcomes,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get recent outcomes", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get recent outcomes: {str(e)}")


@router.get("/format-preview/{location}", summary="Preview formatted alert for location")
async def preview_formatted_alert(
    request: Request,
    location: str,
    hazard_type: str = Query("flood", description="Type of hazard"),
    risk_level: str = Query("HIGH", description="Risk level for preview")
):
    """
    Preview what a formatted alert would look like for a location
    (without actually sending it)
    """
    try:
        # Create mock prediction for preview
        prediction = {
            "location": location,
            "hazard_type": hazard_type,
            "prediction": {
                "risk_score": {"LOW": 0.2, "MODERATE": 0.45, "HIGH": 0.72, "CRITICAL": 0.88}.get(risk_level, 0.5),
                "risk_level": risk_level,
                "hours_to_peak": 18,
                "confidence": 0.75,
                "method_breakdown": {
                    "lstm": 0.70,
                    "pattern_matching": 0.75,
                    "progression": 0.68
                }
            }
        }
        
        # Get location data
        location_data = await _get_location_data(request, location)
        
        # Format alert
        formatted_alert = _alert_formatter.format_prediction(prediction, location_data)
        
        return {
            "success": True,
            "preview": True,
            "alert": formatted_alert.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to preview alert", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to preview alert: {str(e)}")


# ============================================================================
# Helper functions for Phase 3 endpoints
# ============================================================================

async def _get_phase2_prediction(request: Request, location: str, hazard_type: str) -> Dict:
    """
    Get Phase 2 prediction from orchestrator or generate mock prediction
    """
    try:
        # Try to use the orchestrator from app state if available
        if hasattr(request.app.state, 'orchestrator'):
            orchestrator = request.app.state.orchestrator
            # Get weather data for location
            weather_data = await _get_weather_for_location(request, location)
            if weather_data is not None:
                return orchestrator.predict(weather_data, location, hazard_type)
    except Exception as e:
        logger.warning(f"Could not get Phase 2 prediction: {e}, using mock data")
    
    # Return mock prediction for demonstration
    return {
        "location": location,
        "hazard_type": hazard_type,
        "timestamp": datetime.now().isoformat(),
        "prediction": {
            "risk_score": 0.72,
            "risk_level": "HIGH",
            "hours_to_peak": 18,
            "primary_method": "pattern_matching",
            "method_breakdown": {
                "lstm": 0.68,
                "pattern_matching": 0.75,
                "progression": 0.70
            },
            "confidence": 0.75,
            "warnings": [],
            "recommendation": "Prepare emergency response protocols. Monitor conditions closely.",
            "is_anomalous": False
        },
        "forecasts": [
            {"method": "lstm", "risk_score": 0.68, "hours_to_peak": 20, "confidence": 0.70},
            {"method": "pattern_matching", "risk_score": 0.75, "hours_to_peak": 18, "confidence": 0.80},
            {"method": "progression", "risk_score": 0.70, "hours_to_peak": 16, "confidence": 0.65}
        ]
    }


async def _get_location_data(request: Request, location: str) -> Dict:
    """
    Get location data from database or return defaults
    """
    try:
        if hasattr(request.app.state, 'location_service'):
            location_service = request.app.state.location_service
            result = await location_service.search_locations(
                type('Search', (), {'name': location, 'is_active': True, 'limit': 1, 'offset': 0,
                                   'country': None, 'region': None, 'district': None, 
                                   'is_coastal': None, 'tags': None,
                                   'latitude_min': None, 'latitude_max': None,
                                   'longitude_min': None, 'longitude_max': None})()
            )
            if result.get('locations'):
                loc = result['locations'][0]
                return {
                    "name": loc.get('name', location),
                    "latitude": loc.get('latitude', -26.2041),
                    "longitude": loc.get('longitude', 28.0473),
                    "population": loc.get('population', 50000),
                    "region": loc.get('region', 'Unknown'),
                    "is_coastal": loc.get('is_coastal', False)
                }
    except Exception as e:
        logger.warning(f"Could not get location data: {e}, using defaults")
    
    # Return default location data
    return {
        "name": location,
        "latitude": -26.2041,
        "longitude": 28.0473,
        "population": 50000,
        "region": "Gauteng",
        "is_coastal": False
    }


async def _get_weather_for_location(request: Request, location: str):
    """Get weather data for location"""
    try:
        if hasattr(request.app.state, 'weather_service'):
            weather_service = request.app.state.weather_service
            return await weather_service.get_weather_data(location)
    except Exception as e:
        logger.warning(f"Could not get weather data: {e}")
    return None


async def _get_alert_recipients(request: Request, location: str, prediction: Dict) -> List[Dict]:
    """
    Get list of recipients for alert distribution
    
    In production, this would query a database of registered recipients
    """
    # Mock recipients for demonstration
    risk_level = prediction.get('prediction', {}).get('risk_level', 'MODERATE')
    
    recipients = [
        # Authorities always get alerts
        {"type": "authority", "identifier": "eoc@emergency.gov.za"},
        {"type": "authority", "identifier": "disaster.manager@local.gov.za"},
    ]
    
    if risk_level in ["HIGH", "CRITICAL"]:
        # Add more recipients for high risk
        recipients.extend([
            {"type": "critical_facility", "identifier": "hospital@health.gov.za"},
            {"type": "media", "identifier": "news@broadcast.co.za"},
            {"type": "siren_zone", "identifier": f"ZONE-{location.upper()[:3]}"},
        ])
    
    if risk_level == "CRITICAL":
        # Add residential alerts for critical
        recipients.extend([
            {"type": "residential", "identifier": "+27123456789"},
            {"type": "community_leader", "identifier": "+27987654321"},
        ])
    
    return recipients


# ============================================================================
# Additional Phase 3 Endpoints: Actions and Resources
# ============================================================================

class ActionPlanRequest(BaseModel):
    """Request model for generating action plans"""
    location: str = Field(..., description="Location name or ID")
    hazard_type: str = Field(default="flood", description="Type of hazard")
    risk_level: str = Field(default="HIGH", description="Risk level: LOW, MODERATE, HIGH, CRITICAL")
    hours_to_peak: int = Field(default=24, ge=1, le=168, description="Hours until peak conditions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": "Johannesburg",
                "hazard_type": "flood",
                "risk_level": "HIGH",
                "hours_to_peak": 18
            }
        }


class ResourceRequest(BaseModel):
    """Request model for calculating resources"""
    location: str = Field(..., description="Location name or ID")
    hazard_type: str = Field(default="flood", description="Type of hazard")
    risk_level: str = Field(default="HIGH", description="Risk level: LOW, MODERATE, HIGH, CRITICAL")
    population: Optional[int] = Field(None, description="Override population estimate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "location": "Johannesburg",
                "hazard_type": "flood",
                "risk_level": "HIGH",
                "population": 100000
            }
        }


class UpdateInventoryRequest(BaseModel):
    """Request model for updating resource inventory"""
    resource_type: str = Field(..., description="Type of resource")
    quantity: int = Field(..., ge=0, description="New quantity available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "resource_type": "shelter_beds",
                "quantity": 5000
            }
        }


@router.post("/actions/generate", summary="Generate action plan for a location", tags=["Phase 3 - Actions"])
async def generate_action_plan(
    request: Request,
    action_request: ActionPlanRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Generate a comprehensive action plan for emergency response.
    
    Returns:
    - List of specific actions with stakeholders
    - Priority levels and timelines
    - Prerequisites and dependencies
    - Success metrics for each action
    """
    try:
        logger.info("Generating action plan",
                   location=action_request.location,
                   hazard_type=action_request.hazard_type,
                   risk_level=action_request.risk_level)
        
        # Get location data
        location_data = await _get_location_data(request, action_request.location)
        
        # Create prediction-like structure for action generator
        prediction = {
            "location": action_request.location,
            "hazard_type": action_request.hazard_type,
            "prediction": {
                "risk_level": action_request.risk_level,
                "risk_score": {"LOW": 0.2, "MODERATE": 0.45, "HIGH": 0.72, "CRITICAL": 0.88}.get(action_request.risk_level, 0.5),
                "hours_to_peak": action_request.hours_to_peak
            }
        }
        
        # Generate action plan
        action_plan = _action_generator.generate_actions(prediction, location_data)
        
        return {
            "success": True,
            "action_plan": action_plan.to_dict(),
            "summary": {
                "total_actions": len(action_plan.actions),
                "immediate_actions": len([a for a in action_plan.actions if a.priority.value == 0]),
                "critical_path_length": len(action_plan.critical_path),
                "estimated_total_time_min": action_plan.total_estimated_time_min
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to generate action plan", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate action plan: {str(e)}")


@router.get("/actions/by-stakeholder/{location}", summary="Get actions grouped by stakeholder", tags=["Phase 3 - Actions"])
async def get_actions_by_stakeholder(
    request: Request,
    location: str,
    hazard_type: str = Query("flood", description="Type of hazard"),
    risk_level: str = Query("HIGH", description="Risk level")
):
    """
    Get action plan organized by stakeholder for easy assignment.
    """
    try:
        location_data = await _get_location_data(request, location)
        
        prediction = {
            "location": location,
            "hazard_type": hazard_type,
            "prediction": {
                "risk_level": risk_level,
                "risk_score": {"LOW": 0.2, "MODERATE": 0.45, "HIGH": 0.72, "CRITICAL": 0.88}.get(risk_level, 0.5),
                "hours_to_peak": 24
            }
        }
        
        action_plan = _action_generator.generate_actions(prediction, location_data)
        by_stakeholder = action_plan.get_actions_by_stakeholder()
        
        # Convert to serializable format
        result = {}
        for stakeholder, actions in by_stakeholder.items():
            result[stakeholder] = [a.to_dict() for a in actions]
        
        return {
            "success": True,
            "location": location,
            "hazard_type": hazard_type,
            "risk_level": risk_level,
            "actions_by_stakeholder": result,
            "stakeholder_count": len(result),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get actions by stakeholder", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get actions: {str(e)}")


@router.get("/actions/immediate/{location}", summary="Get immediate actions only", tags=["Phase 3 - Actions"])
async def get_immediate_actions(
    request: Request,
    location: str,
    hazard_type: str = Query("flood", description="Type of hazard"),
    risk_level: str = Query("CRITICAL", description="Risk level")
):
    """
    Get only immediate priority actions for urgent situations.
    """
    try:
        location_data = await _get_location_data(request, location)
        
        prediction = {
            "location": location,
            "hazard_type": hazard_type,
            "prediction": {
                "risk_level": risk_level,
                "risk_score": {"LOW": 0.2, "MODERATE": 0.45, "HIGH": 0.72, "CRITICAL": 0.88}.get(risk_level, 0.5),
                "hours_to_peak": 6  # Short time for immediate actions
            }
        }
        
        action_plan = _action_generator.generate_actions(prediction, location_data)
        immediate = _action_generator.get_immediate_actions(action_plan)
        
        return {
            "success": True,
            "location": location,
            "hazard_type": hazard_type,
            "risk_level": risk_level,
            "immediate_actions": [a.to_dict() for a in immediate],
            "count": len(immediate),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get immediate actions", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get immediate actions: {str(e)}")


@router.post("/resources/calculate", summary="Calculate resource requirements", tags=["Phase 3 - Resources"])
async def calculate_resources(
    request: Request,
    resource_request: ResourceRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Calculate resource requirements based on Sphere humanitarian standards.
    
    Returns:
    - Resource quantities needed
    - Cost estimates
    - Current availability
    - Shortage analysis
    - Procurement recommendations
    """
    try:
        logger.info("Calculating resources",
                   location=resource_request.location,
                   hazard_type=resource_request.hazard_type,
                   risk_level=resource_request.risk_level)
        
        # Get location data
        location_data = await _get_location_data(request, resource_request.location)
        
        # Override population if provided
        if resource_request.population:
            location_data["population"] = resource_request.population
        
        # Create prediction-like structure
        prediction = {
            "location": resource_request.location,
            "hazard_type": resource_request.hazard_type,
            "prediction": {
                "risk_level": resource_request.risk_level,
                "risk_score": {"LOW": 0.2, "MODERATE": 0.45, "HIGH": 0.72, "CRITICAL": 0.88}.get(resource_request.risk_level, 0.5),
                "hours_to_peak": 24
            }
        }
        
        # Calculate resources
        resource_report = _resource_calculator.calculate_resources(prediction, location_data)
        
        # Generate procurement plan
        procurement_plan = _resource_calculator.get_procurement_plan(resource_report)
        
        return {
            "success": True,
            "resources": resource_report.to_dict(),
            "procurement_plan": procurement_plan,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to calculate resources", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to calculate resources: {str(e)}")


@router.get("/resources/shortages/{location}", summary="Get critical resource shortages", tags=["Phase 3 - Resources"])
async def get_resource_shortages(
    request: Request,
    location: str,
    hazard_type: str = Query("flood", description="Type of hazard"),
    risk_level: str = Query("HIGH", description="Risk level")
):
    """
    Get critical resource shortages that need immediate procurement.
    """
    try:
        location_data = await _get_location_data(request, location)
        
        prediction = {
            "location": location,
            "hazard_type": hazard_type,
            "prediction": {
                "risk_level": risk_level,
                "risk_score": {"LOW": 0.2, "MODERATE": 0.45, "HIGH": 0.72, "CRITICAL": 0.88}.get(risk_level, 0.5),
                "hours_to_peak": 24
            }
        }
        
        resource_report = _resource_calculator.calculate_resources(prediction, location_data)
        
        return {
            "success": True,
            "location": location,
            "hazard_type": hazard_type,
            "risk_level": risk_level,
            "critical_shortages": resource_report.critical_shortages,
            "total_shortage_cost": resource_report.total_shortage_cost,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get resource shortages", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get shortages: {str(e)}")


@router.put("/resources/inventory", summary="Update resource inventory", tags=["Phase 3 - Resources"])
async def update_resource_inventory(
    request: Request,
    inventory_update: UpdateInventoryRequest,
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Update the available quantity of a resource type.
    
    Used to keep inventory data current for accurate gap analysis.
    """
    try:
        logger.info("Updating inventory",
                   resource_type=inventory_update.resource_type,
                   quantity=inventory_update.quantity)
        
        _resource_calculator.update_inventory(
            inventory_update.resource_type,
            inventory_update.quantity
        )
        
        return {
            "success": True,
            "message": f"Updated {inventory_update.resource_type} to {inventory_update.quantity}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to update inventory", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update inventory: {str(e)}")


@router.get("/resources/categories", summary="Get resources grouped by category", tags=["Phase 3 - Resources"])
async def get_resources_by_category(
    request: Request,
    location: str = Query(..., description="Location name"),
    hazard_type: str = Query("flood", description="Type of hazard"),
    risk_level: str = Query("HIGH", description="Risk level")
):
    """
    Get resource requirements organized by category (shelter, food, water, etc.)
    """
    try:
        location_data = await _get_location_data(request, location)
        
        prediction = {
            "location": location,
            "hazard_type": hazard_type,
            "prediction": {
                "risk_level": risk_level,
                "risk_score": {"LOW": 0.2, "MODERATE": 0.45, "HIGH": 0.72, "CRITICAL": 0.88}.get(risk_level, 0.5),
                "hours_to_peak": 24
            }
        }
        
        resource_report = _resource_calculator.calculate_resources(prediction, location_data)
        by_category = resource_report.get_resources_by_category()
        
        # Convert to serializable format
        result = {}
        for category, resources in by_category.items():
            result[category] = [r.to_dict() for r in resources]
        
        return {
            "success": True,
            "location": location,
            "hazard_type": hazard_type,
            "risk_level": risk_level,
            "resources_by_category": result,
            "category_count": len(result),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get resources by category", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get resources: {str(e)}")


# ============================================================================
# Integration and Distribution Endpoints
# ============================================================================

@router.get("/channels", summary="Get available distribution channels", tags=["Phase 3 - Distribution"])
async def get_available_channels(request: Request):
    """
    Get list of available alert distribution channels.
    """
    try:
        channels = _alert_distributor.get_available_channels()
        
        channel_info = {
            "sms": {"name": "SMS", "description": "Text message alerts", "requires": "phone_number"},
            "email": {"name": "Email", "description": "Email alerts with full details", "requires": "email_address"},
            "dashboard": {"name": "Dashboard", "description": "EOC dashboard notifications", "requires": "dashboard_id"},
            "whatsapp": {"name": "WhatsApp", "description": "WhatsApp Business API", "requires": "phone_number"},
            "phone_tree": {"name": "Phone Tree", "description": "Automated voice calls", "requires": "phone_number"},
            "siren": {"name": "Siren", "description": "Emergency siren activation", "requires": "zone_id"},
            "telegram": {"name": "Telegram", "description": "Telegram bot messages", "requires": "chat_id"}
        }
        
        available = []
        for channel in channels:
            info = channel_info.get(channel, {"name": channel, "description": "Unknown", "requires": "identifier"})
            available.append({"id": channel, **info})
        
        return {
            "success": True,
            "channels": available,
            "count": len(available),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get channels", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get channels: {str(e)}")


@router.get("/integrations/status", summary="Check integration system status", tags=["Phase 3 - Integration"])
async def check_integration_status(request: Request):
    """
    Check connectivity status of all integrated emergency systems.
    """
    try:
        status = await _emergency_integrator.check_system_status()
        
        system_info = {
            "eoc": "Emergency Operations Center Dashboard",
            "disaster_db": "Disaster Management Database",
            "weather_service": "Weather Service API",
            "hospital_network": "Hospital Network Notifications",
            "media_system": "Media Broadcast System",
            "municipal": "Municipal Alert System"
        }
        
        results = []
        for system_id, is_available in status.items():
            results.append({
                "system_id": system_id,
                "name": system_info.get(system_id, system_id),
                "status": "available" if is_available else "unavailable"
            })
        
        return {
            "success": True,
            "systems": results,
            "all_available": all(status.values()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to check integration status", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check status: {str(e)}")


@router.post("/alert/{alert_id}/update", summary="Create update for existing alert", tags=["Phase 3 - Alert Lifecycle"])
async def create_alert_update(
    request: Request,
    alert_id: str,
    new_risk_level: str = Query(..., description="Updated risk level"),
    new_hours_to_peak: int = Query(..., ge=0, description="Updated hours to peak"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Create an update alert for an existing alert (CAP UPDATE message type).
    """
    try:
        # Get original alert from database
        db = get_db(request.app)
        from bson import ObjectId
        
        try:
            original = await db["alerts"].find_one({"_id": ObjectId(alert_id)})
        except:
            original = await db["alerts"].find_one({"identifier": alert_id})
        
        if not original:
            raise HTTPException(status_code=404, detail="Original alert not found")
        
        location = original.get("location", "Unknown")
        hazard_type = original.get("hazard_type", "flood")
        
        # Get location data
        location_data = await _get_location_data(request, location)
        
        # Create new prediction with updated values
        new_prediction = {
            "location": location,
            "hazard_type": hazard_type,
            "prediction": {
                "risk_level": new_risk_level,
                "risk_score": {"LOW": 0.2, "MODERATE": 0.45, "HIGH": 0.72, "CRITICAL": 0.88}.get(new_risk_level, 0.5),
                "hours_to_peak": new_hours_to_peak,
                "confidence": 0.75
            }
        }
        
        # Format original alert
        original_formatted = _alert_formatter.format_prediction(
            {"location": location, "hazard_type": hazard_type, "prediction": original.get("prediction", {})},
            location_data
        )
        
        # Create update alert
        update_alert = _alert_formatter.create_update_alert(
            original_formatted,
            new_prediction,
            location_data
        )
        
        return {
            "success": True,
            "update_alert": update_alert.to_dict(),
            "original_alert_id": alert_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create alert update", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create update: {str(e)}")


@router.post("/alert/{alert_id}/cancel", summary="Cancel an existing alert", tags=["Phase 3 - Alert Lifecycle"])
async def cancel_alert(
    request: Request,
    alert_id: str,
    reason: str = Query("Conditions have improved", description="Reason for cancellation"),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """
    Create a cancellation for an existing alert (CAP CANCEL message type).
    """
    try:
        # Get original alert from database
        db = get_db(request.app)
        from bson import ObjectId
        
        try:
            original = await db["alerts"].find_one({"_id": ObjectId(alert_id)})
        except:
            original = await db["alerts"].find_one({"identifier": alert_id})
        
        if not original:
            raise HTTPException(status_code=404, detail="Original alert not found")
        
        location = original.get("location", "Unknown")
        hazard_type = original.get("hazard_type", "flood")
        
        # Get location data
        location_data = await _get_location_data(request, location)
        
        # Format original alert
        original_formatted = _alert_formatter.format_prediction(
            {"location": location, "hazard_type": hazard_type, "prediction": original.get("prediction", {})},
            location_data
        )
        
        # Create cancel alert
        cancel_alert = _alert_formatter.create_cancel_alert(original_formatted, reason)
        
        # Update original alert status in database
        await db["alerts"].update_one(
            {"_id": original["_id"]},
            {"$set": {"status": "cancelled", "cancelled_at": datetime.now().isoformat(), "cancel_reason": reason}}
        )
        
        return {
            "success": True,
            "cancel_alert": cancel_alert.to_dict(),
            "original_alert_id": alert_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel alert", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to cancel alert: {str(e)}")
