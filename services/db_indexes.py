"""
Database Index Management for Crisis Connect
Ensures optimal database performance
"""

import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = structlog.get_logger(__name__)


async def ensure_service_indexes(db: AsyncIOMotorDatabase):
    """
    Create all necessary indexes for service collections
    
    Args:
        db: MongoDB database instance
    """
    logger.info("Creating database indexes...")
    
    try:
        # Weather data indexes
        await db.weather_data.create_index([
            ("location", 1),
            ("timestamp", -1)
        ], name="location_timestamp_idx")
        
        await db.weather_data.create_index([
            ("timestamp", -1)
        ], name="timestamp_idx")
        
        logger.info("✅ Weather data indexes created")
        
        # Predictions indexes
        await db.predictions.create_index([
            ("location", 1),
            ("risk_score", -1),
            ("timestamp", -1)
        ], name="location_risk_timestamp_idx")
        
        await db.predictions.create_index([
            ("risk_category", 1),
            ("timestamp", -1)
        ], name="risk_category_timestamp_idx")
        
        await db.predictions.create_index([
            ("timestamp", -1)
        ], name="predictions_timestamp_idx")
        
        logger.info("✅ Predictions indexes created")
        
        # Alerts indexes
        await db.alerts.create_index([
            ("location", 1),
            ("timestamp", -1)
        ], name="alerts_location_timestamp_idx")
        
        await db.alerts.create_index([
            ("risk_level", 1),
            ("timestamp", -1)
        ], name="alerts_risk_timestamp_idx")
        
        await db.alerts.create_index([
            ("language", 1),
            ("timestamp", -1)
        ], name="alerts_language_timestamp_idx")
        
        logger.info("✅ Alerts indexes created")
        
        # Locations indexes
        await db.locations.create_index([
            ("name", 1)
        ], name="location_name_idx", unique=True)
        
        await db.locations.create_index([
            ("is_active", 1)
        ], name="location_active_idx")
        
        await db.locations.create_index([
            ("country", 1),
            ("region", 1)
        ], name="location_country_region_idx")
        
        # Geospatial index for location-based queries
        await db.locations.create_index([
            ("coordinates", "2dsphere")
        ], name="location_geo_idx")
        
        logger.info("✅ Locations indexes created")
        
        # Historical events indexes
        await db.historical_flood_events.create_index([
            ("location.name", 1),
            ("start_date", -1)
        ], name="historical_location_date_idx")
        
        await db.historical_flood_events.create_index([
            ("severity", 1),
            ("start_date", -1)
        ], name="historical_severity_date_idx")
        
        await db.historical_flood_events.create_index([
            ("flood_type", 1)
        ], name="historical_flood_type_idx")
        
        await db.historical_flood_events.create_index([
            ("verified", 1)
        ], name="historical_verified_idx")
        
        logger.info("✅ Historical events indexes created")
        
        # Location presets indexes
        await db.location_presets.create_index([
            ("category", 1),
            ("is_public", 1)
        ], name="presets_category_public_idx")
        
        logger.info("✅ Location presets indexes created")
        
        logger.info("🎉 All database indexes created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create indexes: {e}")
        raise


async def drop_all_indexes(db: AsyncIOMotorDatabase):
    """
    Drop all indexes (use with caution!)
    
    Args:
        db: MongoDB database instance
    """
    logger.warning("Dropping all indexes...")
    
    collections = [
        "weather_data",
        "predictions", 
        "alerts",
        "locations",
        "historical_flood_events",
        "location_presets"
    ]
    
    for collection_name in collections:
        try:
            await db[collection_name].drop_indexes()
            logger.info(f"Dropped indexes for {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to drop indexes for {collection_name}: {e}")


async def list_indexes(db: AsyncIOMotorDatabase):
    """
    List all indexes in the database
    
    Args:
        db: MongoDB database instance
        
    Returns:
        Dict of collection names to their indexes
    """
    collections = [
        "weather_data",
        "predictions",
        "alerts", 
        "locations",
        "historical_flood_events",
        "location_presets"
    ]
    
    indexes = {}
    for collection_name in collections:
        try:
            collection_indexes = await db[collection_name].index_information()
            indexes[collection_name] = collection_indexes
        except Exception as e:
            logger.warning(f"Failed to list indexes for {collection_name}: {e}")
    
    return indexes
