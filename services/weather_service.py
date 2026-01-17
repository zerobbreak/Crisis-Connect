"""
Weather data collection and processing service
"""
import asyncio
import time
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import structlog

from services.predict import collect_all_data, generate_risk_scores, calculate_household_resources, compare_with_historical_disasters
from utils.db import get_db
from models.model import WeatherEntry

logger = structlog.get_logger(__name__)


class WeatherService:
    """Service for weather data collection and processing"""
    
    def __init__(self, app):
        self.app = app
        self.db = get_db(app)
    
    async def collect_weather_data(self, locations: Optional[List[Dict[str, Any]]] = None,
                                  location_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Collect weather data for specified locations

        Args:
            locations: List of location dictionaries with lat/lon or name
            location_ids: List of location IDs from database

        Returns:
            Dictionary with collection results
        """
        try:
            logger.info("Starting weather data collection",
                       locations_count=len(locations) if locations else 0,
                       location_ids_count=len(location_ids) if location_ids else 0)
            start_time = time.time()

            # Resolve locations to coordinates
            target_locations = await self._resolve_locations(locations, location_ids)

            # Collect data using the existing function
            df = collect_all_data(target_locations)
            count = len(df)

            if count == 0:
                logger.warning("No weather data collected")
                return {
                    "success": False,
                    "message": "No weather data collected",
                    "count": 0,
                    "duration_seconds": 0
                }

            # Validate and store data
            validated_count = await self._store_weather_data(df)

            duration = time.time() - start_time
            logger.info("Weather data collection completed",
                       total_count=count,
                       validated_count=validated_count,
                       duration_seconds=round(duration, 2))

            return {
                "success": True,
                "message": "Weather data collected successfully",
                "count": count,
                "validated_count": validated_count,
                "duration_seconds": round(duration, 2),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Weather data collection failed", error=str(e), exc_info=True)
            raise

    async def _resolve_locations(self, locations: Optional[List[Dict[str, Any]]] = None,
                                location_ids: Optional[List[str]] = None) -> Optional[Dict]:
        """Resolve locations from various input formats"""
        # Priority 1: Direct location dictionaries with coordinates
        if locations:
            return locations

        # Priority 2: Location IDs from database
        elif location_ids:
            try:
                # Import here to avoid circular imports
                from main import location_service
                coords = await location_service.get_locations_coords(location_ids)
                # Convert to format expected by collect_all_data
                return [
                    {"name": name, "lat": lat, "lon": lon}
                    for name, (lat, lon) in coords.items()
                ]
            except Exception as e:
                logger.error("Failed to resolve location IDs", error=str(e))
                return None

        # Priority 3: None (use defaults in collect_all_data)
        return None
    
    async def _store_weather_data(self, df: pd.DataFrame) -> int:
        """Store validated weather data in database"""
        validated_count = 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for record in df.to_dict(orient="records"):
            try:
                # Validate required fields
                if not all(key in record for key in ["location", "lat", "lon"]):
                    logger.warning("Skipping invalid record", record=record)
                    continue
                
                # Ensure timestamp
                if "timestamp" not in record or not record["timestamp"]:
                    record["timestamp"] = now
                
                # Validate coordinates
                if not (-90 <= record["lat"] <= 90) or not (-180 <= record["lon"] <= 180):
                    logger.warning("Invalid coordinates", record=record)
                    continue
                
                # Store in database
                from pymongo import UpdateOne
                ops = [UpdateOne(
                    {"location": record["location"], "timestamp": record["timestamp"]},
                    {"$set": record},
                    upsert=True
                )]
                await self.db["weather_data"].bulk_write(ops)
                validated_count += 1
                
            except Exception as e:
                logger.warning("Failed to store record", error=str(e), record=record)
                continue
        
        return validated_count
    
    async def get_weather_data(self, 
                             location: Optional[str] = None,
                             limit: int = 100,
                             hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Retrieve weather data from database
        
        Args:
            location: Filter by location name
            limit: Maximum number of records to return
            hours_back: Hours of historical data to retrieve
            
        Returns:
            List of weather data records
        """
        try:
            # Build query
            query = {}
            if location:
                query["location"] = {"$regex": f"^{location}$", "$options": "i"}
            
            # Add time filter
            time_threshold = datetime.now().timestamp() - (hours_back * 3600)
            query["timestamp"] = {"$gte": datetime.fromtimestamp(time_threshold).strftime("%Y-%m-%d %H:%M:%S")}
            
            # Execute query
            cursor = self.db["weather_data"].find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
            results = await cursor.to_list(length=limit)
            
            logger.info("Weather data retrieved", 
                       location=location,
                       count=len(results),
                       hours_back=hours_back)
            
            return results
            
        except Exception as e:
            logger.error("Failed to retrieve weather data", error=str(e), exc_info=True)
            raise
    
    async def process_risk_assessment(self, model, generate_alerts: bool = False) -> Dict[str, Any]:
        """
        Process weather data and generate risk assessments
        
        Args:
            model: Trained ML model
            generate_alerts: Whether to generate alerts for high-risk locations
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info("Starting risk assessment processing")
            start_time = time.time()
            
            # Get existing predictions to protect simulated/high-risk data
            existing_predictions = await self.db["predictions"].find(
                {"composite_risk_score": {"$exists": True}},
                {"location": 1, "composite_risk_score": 1, "scenario": 1}
            ).to_list(length=10000)
            
            # Identify protected locations
            protected_locations = {
                p["location"]: p
                for p in existing_predictions
                if p.get("scenario") and p["scenario"] != "real-time"
                or p.get("composite_risk_score", 0) >= 70
            }
            
            logger.info("Protected locations from overwrite", 
                       count=len(protected_locations),
                       locations=list(protected_locations.keys()))
            
            # Collect new data for unprotected locations
            from services.predict import DISTRICT_COORDS
            locations_to_collect = {
                loc: coords for loc, coords in DISTRICT_COORDS.items()
                if loc not in protected_locations
            }
            
            # Process new data
            if locations_to_collect:
                df_new = collect_all_data(locations_to_collect)
                if not df_new.empty:
                    df_new["scenario"] = "real-time"
                    df_new["scenario"] = "real-time"
                    
                    # Apply historical disaster comparison
                    disaster_risks = []
                    for _, row in df_new.iterrows():
                        weather_dict = row.to_dict()
                        disaster_risk = compare_with_historical_disasters(weather_dict, row['location'])
                        disaster_risks.append(disaster_risk)
                    
                    df_scored = generate_risk_scores(df_new, model)
                    
                    # Override risk score if disaster match is high
                    for i, risk in enumerate(disaster_risks):
                        if risk > 0:
                            # If disaster match found, force high risk score
                            current_score = df_scored.at[i, 'composite_risk_score']
                            df_scored.at[i, 'composite_risk_score'] = max(current_score, risk)
                            logger.warning(f"Risk score boosted to {risk} due to historical disaster match for {df_scored.at[i, 'location']}")
                else:
                    df_scored = pd.DataFrame()
            else:
                df_scored = pd.DataFrame()
            
            # Combine with protected data
            df_protected = pd.DataFrame([
                {k: v for k, v in doc.items() if k != "_id"}
                for doc in protected_locations.values()
            ])
            
            if df_protected.empty:
                df_final = df_scored
            elif df_scored.empty:
                df_final = df_protected
            else:
                df_final = pd.concat([df_scored, df_protected], ignore_index=True)
            
            if df_final.empty:
                logger.warning("No data to process")
                return {
                    "success": False,
                    "message": "No weather data to process",
                    "predictions_count": 0,
                    "alerts_generated": 0
                }
            
            # Store predictions
            predictions_count = await self._store_predictions(df_final)
            
            # Generate alerts if requested
            alerts_generated = []
            if generate_alerts:
                alerts_generated = await self._generate_alerts_from_predictions()
            
            duration = time.time() - start_time
            logger.info("Risk assessment processing completed",
                       predictions_count=predictions_count,
                       alerts_generated=len(alerts_generated),
                       duration_seconds=round(duration, 2))
            
            return {
                "success": True,
                "message": f"Processed {predictions_count} predictions",
                "predictions_count": predictions_count,
                "alerts_generated": len(alerts_generated),
                "duration_seconds": round(duration, 2),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Risk assessment processing failed", error=str(e), exc_info=True)
            raise
    
    async def _store_predictions(self, df: pd.DataFrame) -> int:
        """Store predictions in database"""
        try:
            # Ensure household_resources is properly formatted
            df['household_resources'] = df['household_resources'].apply(
                lambda x: x if isinstance(x, dict) else {}
            )
            
            # Convert to records
            predictions = df.to_dict(orient="records")
            
            # Store in database
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            from pymongo import UpdateOne
            ops = [
                UpdateOne(
                    {"location": p["location"], "timestamp": p.get("timestamp") or now},
                    {"$set": p},
                    upsert=True
                )
                for p in predictions
            ]
            result = await self.db["predictions"].bulk_write(ops)
            
            logger.info("Predictions stored", 
                       inserted=result.upserted_count,
                       modified=result.modified_count)
            
            return len(predictions)
            
        except Exception as e:
            logger.error("Failed to store predictions", error=str(e), exc_info=True)
            raise
    
    async def _generate_alerts_from_predictions(self) -> List[Dict[str, Any]]:
        """Generate alerts from high-risk predictions"""
        try:
            from services.alert_generate import generate_alerts_from_db
            alerts = await generate_alerts_from_db(self.db)
            return alerts
        except Exception as e:
            logger.error("Failed to generate alerts", error=str(e), exc_info=True)
            return []

