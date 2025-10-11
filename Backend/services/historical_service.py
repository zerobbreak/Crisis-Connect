"""
Enhanced Historical Data Service for Crisis Connect
Comprehensive flood event management and analysis
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, UpdateOne
import pandas as pd
import numpy as np
from collections import defaultdict
import structlog

from models.historical_models import (
    HistoricalFloodEvent, HistoricalSummary, FloodEventSearch, 
    FloodEventUpdate, FloodType, FloodSeverityLevel, ImpactMetrics,
    WeatherConditions, GeographicLocation, ResponseMetrics, PredictiveFeatures
)

logger = structlog.get_logger()


class HistoricalDataService:
    """Service for managing comprehensive historical flood data"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collections = {
            'events': 'historical_flood_events',
            'summaries': 'historical_summaries',
            'weather': 'historical_weather',
            'impacts': 'historical_impacts',
            'responses': 'historical_responses'
        }
    
    async def create_event(self, event: HistoricalFloodEvent) -> Dict[str, Any]:
        """Create a new historical flood event"""
        try:
            # Validate event data
            event_dict = event.dict()
            event_dict['created_at'] = datetime.now()
            
            # Store in database
            result = await self.db[self.collections['events']].insert_one(event_dict)
            event_dict['_id'] = str(result.inserted_id)
            
            # Update location summary
            await self._update_location_summary(event.location)
            
            # Create weather record if available
            if event.weather_conditions:
                await self._store_weather_data(event.event_id, event.weather_conditions, event.start_date)
            
            logger.info("Historical flood event created", event_id=event.event_id, location=event.location.name)
            
            return {
                "success": True,
                "message": "Historical flood event created successfully",
                "event_id": event.event_id,
                "document_id": str(result.inserted_id)
            }
            
        except Exception as e:
            logger.error("Failed to create historical event", error=str(e), event_id=event.event_id)
            raise
    
    async def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific historical flood event"""
        try:
            event = await self.db[self.collections['events']].find_one({"event_id": event_id})
            if event:
                event['_id'] = str(event['_id'])
            return event
            
        except Exception as e:
            logger.error("Failed to get historical event", error=str(e), event_id=event_id)
            return None
    
    async def search_events(self, search_criteria: FloodEventSearch) -> Dict[str, Any]:
        """Search historical flood events with advanced filtering"""
        try:
            # Build MongoDB query
            query = {}
            
            # Location filters
            if search_criteria.location_name:
                query["location.name"] = {"$regex": search_criteria.location_name, "$options": "i"}
            if search_criteria.district:
                query["location.district"] = {"$regex": search_criteria.district, "$options": "i"}
            if search_criteria.province:
                query["location.province"] = {"$regex": search_criteria.province, "$options": "i"}
            
            # Date filters
            if search_criteria.start_date_from or search_criteria.start_date_to:
                date_filter = {}
                if search_criteria.start_date_from:
                    date_filter["$gte"] = search_criteria.start_date_from
                if search_criteria.start_date_to:
                    date_filter["$lte"] = search_criteria.start_date_to
                query["start_date"] = date_filter
            
            # Severity and type filters
            if search_criteria.severity_levels:
                query["severity"] = {"$in": search_criteria.severity_levels}
            if search_criteria.flood_types:
                query["flood_type"] = {"$in": search_criteria.flood_types}
            
            # Impact thresholds
            if search_criteria.min_deaths:
                query["impacts.deaths"] = {"$gte": search_criteria.min_deaths}
            if search_criteria.min_damage_usd:
                query["impacts.total_economic_impact_usd"] = {"$gte": search_criteria.min_damage_usd}
            
            # Data quality filters
            if search_criteria.min_data_quality:
                quality_order = ["poor", "fair", "good", "excellent"]
                min_index = quality_order.index(search_criteria.min_data_quality)
                query["data_quality"] = {"$in": quality_order[min_index:]}
            
            if search_criteria.verified_only:
                query["verified"] = True
            
            # Build sort criteria
            sort_field = search_criteria.sort_by or "start_date"
            sort_direction = DESCENDING if search_criteria.sort_order == "desc" else ASCENDING
            sort_criteria = [(sort_field, sort_direction)]
            
            # Execute query with pagination
            cursor = self.db[self.collections['events']].find(query).sort(sort_criteria)
            total_count = await self.db[self.collections['events']].count_documents(query)
            
            # Apply pagination
            if search_criteria.offset:
                cursor = cursor.skip(search_criteria.offset)
            cursor = cursor.limit(search_criteria.limit)
            
            events = await cursor.to_list(length=search_criteria.limit)
            
            # Convert ObjectIds to strings
            for event in events:
                event['_id'] = str(event['_id'])
            
            return {
                "success": True,
                "events": events,
                "total_count": total_count,
                "returned_count": len(events),
                "search_criteria": search_criteria.dict()
            }
            
        except Exception as e:
            logger.error("Failed to search historical events", error=str(e))
            raise
    
    async def update_event(self, event_id: str, update_data: FloodEventUpdate) -> Dict[str, Any]:
        """Update an existing historical flood event"""
        try:
            # Convert update data to dictionary, excluding None values
            update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
            update_dict['last_updated'] = datetime.now()
            
            # Update the event
            result = await self.db[self.collections['events']].update_one(
                {"event_id": event_id},
                {"$set": update_dict}
            )
            
            if result.matched_count == 0:
                return {
                    "success": False,
                    "message": f"Event {event_id} not found"
                }
            
            # Update location summary if location-related fields changed
            if any(field in update_dict for field in ['severity', 'impacts']):
                event = await self.get_event(event_id)
                if event:
                    location = GeographicLocation(**event['location'])
                    await self._update_location_summary(location)
            
            logger.info("Historical flood event updated", event_id=event_id)
            
            return {
                "success": True,
                "message": "Historical flood event updated successfully",
                "modified_count": result.modified_count
            }
            
        except Exception as e:
            logger.error("Failed to update historical event", error=str(e), event_id=event_id)
            raise
    
    async def delete_event(self, event_id: str) -> Dict[str, Any]:
        """Delete a historical flood event"""
        try:
            # Get event before deletion for summary update
            event = await self.get_event(event_id)
            
            # Delete the event
            result = await self.db[self.collections['events']].delete_one({"event_id": event_id})
            
            if result.deleted_count == 0:
                return {
                    "success": False,
                    "message": f"Event {event_id} not found"
                }
            
            # Update location summary
            if event:
                location = GeographicLocation(**event['location'])
                await self._update_location_summary(location)
            
            logger.info("Historical flood event deleted", event_id=event_id)
            
            return {
                "success": True,
                "message": "Historical flood event deleted successfully"
            }
            
        except Exception as e:
            logger.error("Failed to delete historical event", error=str(e), event_id=event_id)
            raise
    
    async def get_location_summary(self, location_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary for a specific location"""
        try:
            summary = await self.db[self.collections['summaries']].find_one(
                {"location.name": location_name}
            )
            
            if summary:
                summary['_id'] = str(summary['_id'])
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get location summary", error=str(e), location=location_name)
            return None
    
    async def get_analytics(self, 
                          location: Optional[str] = None,
                          start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> Dict[str, Any]:
        """Get comprehensive analytics for historical flood data"""
        try:
            # Build query
            query = {}
            if location:
                query["location.name"] = {"$regex": location, "$options": "i"}
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query["start_date"] = date_filter
            
            # Get all matching events
            events = await self.db[self.collections['events']].find(query).to_list(length=None)
            
            if not events:
                return {"success": True, "analytics": {}, "message": "No events found for criteria"}
            
            # Calculate analytics
            analytics = self._calculate_analytics(events)
            
            return {
                "success": True,
                "analytics": analytics,
                "total_events": len(events),
                "query_criteria": query
            }
            
        except Exception as e:
            logger.error("Failed to get analytics", error=str(e))
            raise
    
    async def import_legacy_data(self, file_path: str) -> Dict[str, Any]:
        """Import and convert legacy Excel data to new format"""
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            imported_count = 0
            errors = []
            
            for _, row in df.iterrows():
                try:
                    # Convert legacy data to new format
                    event = self._convert_legacy_event(row)
                    
                    # Create event
                    result = await self.create_event(event)
                    if result["success"]:
                        imported_count += 1
                    
                except Exception as e:
                    errors.append(f"Row {row.name}: {str(e)}")
                    continue
            
            logger.info("Legacy data import completed", imported=imported_count, errors=len(errors))
            
            return {
                "success": True,
                "imported_count": imported_count,
                "error_count": len(errors),
                "errors": errors[:10]  # Limit error details
            }
            
        except Exception as e:
            logger.error("Failed to import legacy data", error=str(e))
            raise
    
    async def _update_location_summary(self, location: GeographicLocation):
        """Update summary statistics for a location"""
        try:
            # Get all events for this location
            events = await self.db[self.collections['events']].find({
                "location.name": location.name
            }).to_list(length=None)
            
            if not events:
                return
            
            # Calculate summary
            summary = self._calculate_location_summary(location, events)
            
            # Update or create summary
            await self.db[self.collections['summaries']].update_one(
                {"location.name": location.name},
                {"$set": summary.dict()},
                upsert=True
            )
            
        except Exception as e:
            logger.error("Failed to update location summary", error=str(e), location=location.name)
    
    async def _store_weather_data(self, event_id: str, weather: WeatherConditions, event_date: date):
        """Store weather data associated with an event"""
        try:
            weather_dict = weather.dict()
            weather_dict['event_id'] = event_id
            weather_dict['event_date'] = event_date
            weather_dict['stored_at'] = datetime.now()
            
            await self.db[self.collections['weather']].insert_one(weather_dict)
            
        except Exception as e:
            logger.error("Failed to store weather data", error=str(e), event_id=event_id)
    
    def _calculate_analytics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive analytics from events"""
        if not events:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(events)
        
        analytics = {
            "overview": {
                "total_events": len(events),
                "date_range": {
                    "earliest": df['start_date'].min().isoformat() if 'start_date' in df.columns else None,
                    "latest": df['start_date'].max().isoformat() if 'start_date' in df.columns else None
                }
            },
            "severity_distribution": {},
            "flood_type_distribution": {},
            "temporal_patterns": {},
            "impact_statistics": {},
            "geographic_analysis": {}
        }
        
        # Severity distribution
        if 'severity' in df.columns:
            analytics["severity_distribution"] = df['severity'].value_counts().to_dict()
        
        # Flood type distribution
        if 'flood_type' in df.columns:
            analytics["flood_type_distribution"] = df['flood_type'].value_counts().to_dict()
        
        # Temporal patterns
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['month'] = df['start_date'].dt.month
            df['year'] = df['start_date'].dt.year
            
            analytics["temporal_patterns"] = {
                "by_month": df['month'].value_counts().sort_index().to_dict(),
                "by_year": df['year'].value_counts().sort_index().to_dict()
            }
        
        # Impact statistics
        if 'impacts' in df.columns:
            impacts_data = []
            for impact in df['impacts']:
                if isinstance(impact, dict):
                    impacts_data.append(impact)
            
            if impacts_data:
                impacts_df = pd.DataFrame(impacts_data)
                
                analytics["impact_statistics"] = {
                    "total_deaths": impacts_df['deaths'].sum() if 'deaths' in impacts_df.columns else 0,
                    "total_injuries": impacts_df['injuries'].sum() if 'injuries' in impacts_df.columns else 0,
                    "total_displaced": impacts_df['displaced_persons'].sum() if 'displaced_persons' in impacts_df.columns else 0,
                    "total_damage_usd": impacts_df['total_economic_impact_usd'].sum() if 'total_economic_impact_usd' in impacts_df.columns else 0,
                    "average_damage_per_event": impacts_df['total_economic_impact_usd'].mean() if 'total_economic_impact_usd' in impacts_df.columns else 0
                }
        
        # Geographic analysis
        if 'location' in df.columns:
            locations = []
            for loc in df['location']:
                if isinstance(loc, dict):
                    locations.append(loc)
            
            if locations:
                locations_df = pd.DataFrame(locations)
                
                analytics["geographic_analysis"] = {
                    "unique_locations": len(locations_df),
                    "provinces": locations_df['province'].value_counts().to_dict() if 'province' in locations_df.columns else {},
                    "districts": locations_df['district'].value_counts().to_dict() if 'district' in locations_df.columns else {}
                }
        
        return analytics
    
    def _calculate_location_summary(self, location: GeographicLocation, events: List[Dict[str, Any]]) -> HistoricalSummary:
        """Calculate summary for a specific location"""
        if not events:
            return HistoricalSummary(location=location, total_events=0)
        
        df = pd.DataFrame(events)
        
        # Count events by type and severity
        events_by_type = df['flood_type'].value_counts().to_dict() if 'flood_type' in df.columns else {}
        events_by_severity = df['severity'].value_counts().to_dict() if 'severity' in df.columns else {}
        
        # Temporal patterns
        if 'start_date' in df.columns:
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['month'] = df['start_date'].dt.month
            df['year'] = df['start_date'].dt.year
            
            events_by_month = df['month'].value_counts().sort_index().to_dict()
            events_by_year = df['year'].value_counts().sort_index().to_dict()
        else:
            events_by_month = {}
            events_by_year = {}
        
        # Impact statistics
        total_deaths = 0
        total_injuries = 0
        total_displaced = 0
        total_damage = 0.0
        
        if 'impacts' in df.columns:
            for impact in df['impacts']:
                if isinstance(impact, dict):
                    total_deaths += impact.get('deaths', 0)
                    total_injuries += impact.get('injuries', 0)
                    total_displaced += impact.get('displaced_persons', 0)
                    total_damage += impact.get('total_economic_impact_usd', 0)
        
        # Calculate risk indicators
        years_span = len(events_by_year) if events_by_year else 1
        flood_frequency = len(events) / max(years_span, 1)
        
        # Determine risk trend (simplified)
        if len(events_by_year) >= 3:
            recent_years = sorted(events_by_year.keys())[-3:]
            recent_avg = sum(events_by_year[year] for year in recent_years) / 3
            earlier_years = sorted(events_by_year.keys())[:-3] if len(events_by_year) > 3 else []
            
            if earlier_years:
                earlier_avg = sum(events_by_year[year] for year in earlier_years) / len(earlier_years)
                if recent_avg > earlier_avg * 1.2:
                    risk_trend = "increasing"
                elif recent_avg < earlier_avg * 0.8:
                    risk_trend = "decreasing"
                else:
                    risk_trend = "stable"
            else:
                risk_trend = "stable"
        else:
            risk_trend = None
        
        return HistoricalSummary(
            location=location,
            total_events=len(events),
            events_by_type=events_by_type,
            events_by_severity=events_by_severity,
            events_by_month=events_by_month,
            events_by_year=events_by_year,
            total_deaths=total_deaths,
            total_injuries=total_injuries,
            total_displaced=total_displaced,
            total_property_damage_usd=total_damage,
            flood_frequency_per_year=flood_frequency,
            risk_trend=risk_trend,
            data_completeness_percent=85.0,  # Simplified calculation
            last_event_date=max(pd.to_datetime(df['start_date']).dt.date) if 'start_date' in df.columns else None,
            first_event_date=min(pd.to_datetime(df['start_date']).dt.date) if 'start_date' in df.columns else None
        )
    
    def _convert_legacy_event(self, row: pd.Series) -> HistoricalFloodEvent:
        """Convert legacy Excel data to new HistoricalFloodEvent format"""
        # This is a simplified conversion - would need to be customized based on actual Excel structure
        event_id = f"LEGACY_{row.get('DisNo.', '')}"
        
        # Create location
        location = GeographicLocation(
            name=row.get('Location', 'Unknown'),
            latitude=float(row.get('Latitude', 0)),
            longitude=float(row.get('Longitude', 0)),
            district=row.get('Admin Units', ''),
            province=row.get('Region', ''),
            country="South Africa"
        )
        
        # Create impacts
        impacts = ImpactMetrics(
            deaths=int(row.get('Total Deaths', 0) or 0),
            injuries=int(row.get('No. Injured', 0) or 0),
            displaced_persons=int(row.get('No. Affected', 0) or 0),
            total_economic_impact_usd=float(row.get('Total Damage (\'000 US$)', 0) or 0) * 1000
        )
        
        # Determine flood type and severity (simplified mapping)
        disaster_type = row.get('Disaster Type', '').lower()
        if 'flood' in disaster_type:
            flood_type = FloodType.RIVER_FLOOD
        elif 'flash' in disaster_type:
            flood_type = FloodType.FLASH_FLOOD
        else:
            flood_type = FloodType.RIVER_FLOOD
        
        # Determine severity based on deaths and damage
        deaths = impacts.deaths
        damage = impacts.total_economic_impact_usd
        
        if deaths >= 100 or damage >= 100000000:  # 100M USD
            severity = FloodSeverityLevel.CATASTROPHIC
        elif deaths >= 20 or damage >= 10000000:  # 10M USD
            severity = FloodSeverityLevel.EXTREME
        elif deaths >= 5 or damage >= 1000000:  # 1M USD
            severity = FloodSeverityLevel.SEVERE
        elif deaths >= 1 or damage >= 100000:  # 100K USD
            severity = FloodSeverityLevel.MODERATE
        else:
            severity = FloodSeverityLevel.MINOR
        
        # Parse dates
        start_date = pd.to_datetime(f"{row.get('Start Year', 2000)}-{row.get('Start Month', 1)}-{row.get('Start Day', 1)}").date()
        
        return HistoricalFloodEvent(
            event_id=event_id,
            name=row.get('Event Name', ''),
            start_date=start_date,
            flood_type=flood_type,
            severity=severity,
            location=location,
            impacts=impacts,
            data_source="Legacy Excel Import",
            data_quality="fair",
            verified=False
        )
