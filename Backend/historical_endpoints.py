"""
Enhanced Historical Data API Endpoints
Comprehensive flood event management endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Dict, Any, Optional
from datetime import date, datetime
import structlog

from models.historical_models import (
    HistoricalFloodEvent, HistoricalSummary, FloodEventSearch, 
    FloodEventUpdate, FloodType, FloodSeverityLevel
)
from services.historical_service import HistoricalDataService
from utils.db import get_db
from main import rate_limit, verify_api_key

logger = structlog.get_logger()

# Create router for historical endpoints
router = APIRouter(prefix="/historical", tags=["Historical Data"])

# Global service instance
historical_service: Optional[HistoricalDataService] = None


def get_historical_service():
    """Get historical data service instance"""
    global historical_service
    if historical_service is None:
        db = get_db()
        historical_service = HistoricalDataService(db)
    return historical_service


@router.post("/events", summary="Create a new historical flood event")
async def create_historical_event(
    event: HistoricalFloodEvent,
    service: HistoricalDataService = Depends(get_historical_service),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Create a new comprehensive historical flood event record"""
    try:
        result = await service.create_event(event)
        return result
    except Exception as e:
        logger.error("Failed to create historical event", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create event: {str(e)}")


@router.get("/events/{event_id}", summary="Get a specific historical flood event")
async def get_historical_event(
    event_id: str = Path(..., description="Unique event identifier"),
    service: HistoricalDataService = Depends(get_historical_service)
):
    """Get detailed information about a specific historical flood event"""
    try:
        event = await service.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
        return {"success": True, "event": event}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get historical event", error=str(e), event_id=event_id)
        raise HTTPException(status_code=500, detail=f"Failed to get event: {str(e)}")


@router.post("/events/search", summary="Search historical flood events")
async def search_historical_events(
    search_criteria: FloodEventSearch,
    service: HistoricalDataService = Depends(get_historical_service),
    _: bool = Depends(rate_limit)
):
    """Search historical flood events with advanced filtering and pagination"""
    try:
        result = await service.search_events(search_criteria)
        return result
    except Exception as e:
        logger.error("Failed to search historical events", error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.put("/events/{event_id}", summary="Update a historical flood event")
async def update_historical_event(
    event_id: str = Path(..., description="Unique event identifier"),
    update_data: FloodEventUpdate,
    service: HistoricalDataService = Depends(get_historical_service),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Update an existing historical flood event"""
    try:
        result = await service.update_event(event_id, update_data)
        return result
    except Exception as e:
        logger.error("Failed to update historical event", error=str(e), event_id=event_id)
        raise HTTPException(status_code=500, detail=f"Failed to update event: {str(e)}")


@router.delete("/events/{event_id}", summary="Delete a historical flood event")
async def delete_historical_event(
    event_id: str = Path(..., description="Unique event identifier"),
    service: HistoricalDataService = Depends(get_historical_service),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Delete a historical flood event"""
    try:
        result = await service.delete_event(event_id)
        return result
    except Exception as e:
        logger.error("Failed to delete historical event", error=str(e), event_id=event_id)
        raise HTTPException(status_code=500, detail=f"Failed to delete event: {str(e)}")


@router.get("/locations/{location_name}/summary", summary="Get location summary")
async def get_location_summary(
    location_name: str = Path(..., description="Location name"),
    service: HistoricalDataService = Depends(get_historical_service)
):
    """Get comprehensive summary statistics for a specific location"""
    try:
        summary = await service.get_location_summary(location_name)
        if not summary:
            raise HTTPException(status_code=404, detail=f"Summary for location {location_name} not found")
        return {"success": True, "summary": summary}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get location summary", error=str(e), location=location_name)
        raise HTTPException(status_code=500, detail=f"Failed to get location summary: {str(e)}")


@router.get("/analytics", summary="Get historical flood analytics")
async def get_historical_analytics(
    location: Optional[str] = Query(None, description="Filter by location name"),
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter"),
    service: HistoricalDataService = Depends(get_historical_service),
    _: bool = Depends(rate_limit)
):
    """Get comprehensive analytics for historical flood data"""
    try:
        analytics = await service.get_analytics(location, start_date, end_date)
        return analytics
    except Exception as e:
        logger.error("Failed to get historical analytics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


@router.post("/import/legacy", summary="Import legacy Excel data")
async def import_legacy_data(
    file_path: str = Query(..., description="Path to legacy Excel file"),
    service: HistoricalDataService = Depends(get_historical_service),
    _: bool = Depends(rate_limit),
    api_key: bool = Depends(verify_api_key)
):
    """Import and convert legacy Excel data to new comprehensive format"""
    try:
        result = await service.import_legacy_data(file_path)
        return result
    except Exception as e:
        logger.error("Failed to import legacy data", error=str(e))
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.get("/flood-types", summary="Get available flood types")
async def get_flood_types():
    """Get list of available flood types"""
    return {
        "success": True,
        "flood_types": [ft.value for ft in FloodType],
        "descriptions": {
            "flash_flood": "Sudden, rapid flooding caused by intense rainfall",
            "river_flood": "Overflow of river banks due to excessive water",
            "coastal_flood": "Flooding from storm surge or high tides",
            "urban_flood": "Flooding in urban areas due to poor drainage",
            "dam_break": "Flooding caused by dam or levee failure",
            "storm_surge": "Coastal flooding from tropical storms",
            "seasonal_flood": "Regular seasonal flooding patterns"
        }
    }


@router.get("/severity-levels", summary="Get available severity levels")
async def get_severity_levels():
    """Get list of available severity levels"""
    return {
        "success": True,
        "severity_levels": [sl.value for sl in FloodSeverityLevel],
        "descriptions": {
            "minor": "Minimal impact, local flooding",
            "moderate": "Significant local impact",
            "severe": "Major regional impact",
            "extreme": "Widespread catastrophic impact",
            "catastrophic": "Nationwide disaster impact"
        }
    }


@router.get("/statistics", summary="Get overall historical statistics")
async def get_historical_statistics(
    service: HistoricalDataService = Depends(get_historical_service),
    _: bool = Depends(rate_limit)
):
    """Get overall statistics about historical flood data"""
    try:
        # Get total counts
        db = get_db()
        
        total_events = await db["historical_flood_events"].count_documents({})
        verified_events = await db["historical_flood_events"].count_documents({"verified": True})
        total_locations = len(await db["historical_flood_events"].distinct("location.name"))
        
        # Get date range
        earliest_event = await db["historical_flood_events"].find_one(
            {}, sort=[("start_date", 1)]
        )
        latest_event = await db["historical_flood_events"].find_one(
            {}, sort=[("start_date", -1)]
        )
        
        # Get severity distribution
        severity_pipeline = [
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        severity_dist = await db["historical_flood_events"].aggregate(severity_pipeline).to_list(None)
        
        # Get flood type distribution
        type_pipeline = [
            {"$group": {"_id": "$flood_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        type_dist = await db["historical_flood_events"].aggregate(type_pipeline).to_list(None)
        
        return {
            "success": True,
            "statistics": {
                "total_events": total_events,
                "verified_events": verified_events,
                "verification_rate": round((verified_events / total_events * 100) if total_events > 0 else 0, 2),
                "total_locations": total_locations,
                "date_range": {
                    "earliest": earliest_event["start_date"].isoformat() if earliest_event else None,
                    "latest": latest_event["start_date"].isoformat() if latest_event else None
                },
                "severity_distribution": {item["_id"]: item["count"] for item in severity_dist},
                "flood_type_distribution": {item["_id"]: item["count"] for item in type_dist}
            }
        }
        
    except Exception as e:
        logger.error("Failed to get historical statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/recent", summary="Get recent flood events")
async def get_recent_events(
    limit: int = Query(10, ge=1, le=100, description="Number of recent events to return"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    service: HistoricalDataService = Depends(get_historical_service)
):
    """Get recent flood events within specified time period"""
    try:
        from datetime import timedelta
        start_date = date.today() - timedelta(days=days)
        
        search_criteria = FloodEventSearch(
            start_date_from=start_date,
            limit=limit,
            sort_by="start_date",
            sort_order="desc"
        )
        
        result = await service.search_events(search_criteria)
        return result
        
    except Exception as e:
        logger.error("Failed to get recent events", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get recent events: {str(e)}")


@router.get("/high-impact", summary="Get high-impact flood events")
async def get_high_impact_events(
    min_deaths: int = Query(10, ge=0, description="Minimum number of deaths"),
    min_damage: float = Query(1000000, ge=0, description="Minimum damage in USD"),
    limit: int = Query(20, ge=1, le=100, description="Number of events to return"),
    service: HistoricalDataService = Depends(get_historical_service)
):
    """Get high-impact flood events based on casualties and damage"""
    try:
        search_criteria = FloodEventSearch(
            min_deaths=min_deaths,
            min_damage_usd=min_damage,
            limit=limit,
            sort_by="deaths",
            sort_order="desc"
        )
        
        result = await service.search_events(search_criteria)
        return result
        
    except Exception as e:
        logger.error("Failed to get high-impact events", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get high-impact events: {str(e)}")


@router.get("/patterns", summary="Analyze flood patterns")
async def analyze_flood_patterns(
    location: Optional[str] = Query(None, description="Filter by location"),
    years_back: int = Query(10, ge=1, le=50, description="Years of data to analyze"),
    service: HistoricalDataService = Depends(get_historical_service)
):
    """Analyze temporal and spatial patterns in flood events"""
    try:
        from datetime import timedelta
        start_date = date.today() - timedelta(days=years_back * 365)
        
        analytics = await service.get_analytics(location, start_date, None)
        
        # Add pattern analysis
        if analytics.get("success") and analytics.get("analytics"):
            patterns = _analyze_patterns(analytics["analytics"])
            analytics["patterns"] = patterns
        
        return analytics
        
    except Exception as e:
        logger.error("Failed to analyze flood patterns", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to analyze patterns: {str(e)}")


def _analyze_patterns(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze patterns in flood data"""
    patterns = {
        "seasonal_patterns": {},
        "trend_analysis": {},
        "risk_correlations": {},
        "predictive_insights": []
    }
    
    # Analyze seasonal patterns
    if "temporal_patterns" in analytics and "by_month" in analytics["temporal_patterns"]:
        monthly_data = analytics["temporal_patterns"]["by_month"]
        
        # Find peak months
        peak_months = sorted(monthly_data.items(), key=lambda x: x[1], reverse=True)[:3]
        patterns["seasonal_patterns"]["peak_months"] = [
            {"month": month, "events": count} for month, count in peak_months
        ]
        
        # Calculate seasonality score
        total_events = sum(monthly_data.values())
        if total_events > 0:
            variance = sum((count - total_events/12)**2 for count in monthly_data.values()) / 12
            patterns["seasonal_patterns"]["seasonality_score"] = round(variance / (total_events/12), 2)
    
    # Analyze trends
    if "temporal_patterns" in analytics and "by_year" in analytics["temporal_patterns"]:
        yearly_data = analytics["temporal_patterns"]["by_year"]
        if len(yearly_data) >= 3:
            years = sorted(yearly_data.keys())
            counts = [yearly_data[year] for year in years]
            
            # Simple trend calculation
            if len(counts) >= 2:
                trend = (counts[-1] - counts[0]) / len(counts)
                patterns["trend_analysis"]["trend_direction"] = "increasing" if trend > 0 else "decreasing"
                patterns["trend_analysis"]["trend_strength"] = abs(trend)
    
    # Risk correlations
    if "severity_distribution" in analytics and "flood_type_distribution" in analytics:
        patterns["risk_correlations"]["most_common_severity"] = max(
            analytics["severity_distribution"].items(), 
            key=lambda x: x[1]
        )[0] if analytics["severity_distribution"] else None
        
        patterns["risk_correlations"]["most_common_type"] = max(
            analytics["flood_type_distribution"].items(), 
            key=lambda x: x[1]
        )[0] if analytics["flood_type_distribution"] else None
    
    # Predictive insights
    patterns["predictive_insights"] = [
        "Historical data shows seasonal patterns that can inform early warning systems",
        "Severity distribution helps prioritize response resources",
        "Temporal trends indicate changing flood risk patterns over time"
    ]
    
    return patterns
