#!/usr/bin/env python3
"""
Historical Data Migration Script
Upgrade existing historical data to new comprehensive format
"""

import asyncio
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from models.historical_models import (
    HistoricalFloodEvent, FloodType, FloodSeverityLevel, 
    GeographicLocation, ImpactMetrics, WeatherConditions
)
from services.historical_service import HistoricalDataService
from utils.db import init_mongo, close_mongo, get_db
import structlog

logger = structlog.get_logger()


async def migrate_legacy_data():
    """Migrate legacy Excel data to new comprehensive format"""
    
    print("🔄 Starting historical data migration...")
    
    try:
        # Initialize database connection
        await init_mongo()
        db = get_db()
        service = HistoricalDataService(db)
        
        # Check if legacy data exists
        legacy_file = Path("data/data_disaster.xlsx")
        if not legacy_file.exists():
            print("❌ Legacy data file not found: data/data_disaster.xlsx")
            return False
        
        print(f"📊 Reading legacy data from {legacy_file}")
        df = pd.read_excel(legacy_file)
        print(f"✅ Loaded {len(df)} records from legacy data")
        
        # Show column information
        print("\n📋 Legacy data columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col}")
        
        # Preview first few rows
        print("\n📖 Sample data preview:")
        print(df.head(3).to_string())
        
        # Migration statistics
        migrated_count = 0
        error_count = 0
        errors = []
        
        print(f"\n🚀 Starting migration of {len(df)} records...")
        
        for index, row in df.iterrows():
            try:
                # Create event ID
                event_id = f"MIGRATED_{row.get('DisNo.', f'UNKNOWN_{index}')}"
                
                # Create location
                location = GeographicLocation(
                    name=str(row.get('Location', 'Unknown')).strip(),
                    latitude=float(row.get('Latitude', 0)) if pd.notna(row.get('Latitude')) else 0.0,
                    longitude=float(row.get('Longitude', 0)) if pd.notna(row.get('Longitude')) else 0.0,
                    district=str(row.get('Admin Units', '')).strip() if pd.notna(row.get('Admin Units')) else None,
                    province=str(row.get('Region', '')).strip() if pd.notna(row.get('Region')) else None,
                    country="South Africa"
                )
                
                # Parse dates
                start_year = int(row.get('Start Year', 2000)) if pd.notna(row.get('Start Year')) else 2000
                start_month = int(row.get('Start Month', 1)) if pd.notna(row.get('Start Month')) else 1
                start_day = int(row.get('Start Day', 1)) if pd.notna(row.get('Start Day')) else 1
                
                try:
                    start_date = date(start_year, start_month, start_day)
                except ValueError:
                    start_date = date(start_year, 1, 1)  # Fallback to January 1st
                
                # Determine flood type
                disaster_type = str(row.get('Disaster Type', '')).lower()
                if 'flash' in disaster_type:
                    flood_type = FloodType.FLASH_FLOOD
                elif 'coastal' in disaster_type:
                    flood_type = FloodType.COASTAL_FLOOD
                elif 'urban' in disaster_type:
                    flood_type = FloodType.URBAN_FLOOD
                elif 'dam' in disaster_type:
                    flood_type = FloodType.DAM_BREAK
                elif 'storm' in disaster_type:
                    flood_type = FloodType.STORM_SURGE
                else:
                    flood_type = FloodType.RIVER_FLOOD
                
                # Create impact metrics
                deaths = int(row.get('Total Deaths', 0)) if pd.notna(row.get('Total Deaths')) else 0
                injuries = int(row.get('No. Injured', 0)) if pd.notna(row.get('No. Injured')) else 0
                affected = int(row.get('No. Affected', 0)) if pd.notna(row.get('No. Affected')) else 0
                homeless = int(row.get('No. Homeless', 0)) if pd.notna(row.get('No. Homeless')) else 0
                
                # Calculate damage (convert from thousands USD to USD)
                total_damage = 0.0
                if pd.notna(row.get('Total Damage (\'000 US$)')):
                    total_damage = float(row.get('Total Damage (\'000 US$)', 0)) * 1000
                
                impacts = ImpactMetrics(
                    deaths=deaths,
                    injuries=injuries,
                    displaced_persons=affected,
                    evacuated_persons=homeless,
                    total_economic_impact_usd=total_damage
                )
                
                # Determine severity based on impact
                if deaths >= 100 or total_damage >= 100000000:
                    severity = FloodSeverityLevel.CATASTROPHIC
                elif deaths >= 20 or total_damage >= 10000000:
                    severity = FloodSeverityLevel.EXTREME
                elif deaths >= 5 or total_damage >= 1000000:
                    severity = FloodSeverityLevel.SEVERE
                elif deaths >= 1 or total_damage >= 100000:
                    severity = FloodSeverityLevel.MODERATE
                else:
                    severity = FloodSeverityLevel.MINOR
                
                # Create the historical event
                event = HistoricalFloodEvent(
                    event_id=event_id,
                    name=str(row.get('Event Name', '')).strip() if pd.notna(row.get('Event Name')) else None,
                    start_date=start_date,
                    flood_type=flood_type,
                    severity=severity,
                    location=location,
                    impacts=impacts,
                    data_source="Legacy Excel Migration",
                    data_quality="fair",
                    verified=False,
                    description=f"Migrated from legacy data. Original disaster type: {row.get('Disaster Type', 'Unknown')}"
                )
                
                # Save the event
                result = await service.create_event(event)
                if result["success"]:
                    migrated_count += 1
                    if migrated_count % 10 == 0:
                        print(f"✅ Migrated {migrated_count} events...")
                else:
                    error_count += 1
                    errors.append(f"Row {index}: {result.get('message', 'Unknown error')}")
                
            except Exception as e:
                error_count += 1
                error_msg = f"Row {index}: {str(e)}"
                errors.append(error_msg)
                logger.error("Migration error", error=str(e), row=index)
                continue
        
        # Print migration summary
        print(f"\n🎉 Migration completed!")
        print(f"✅ Successfully migrated: {migrated_count} events")
        print(f"❌ Errors encountered: {error_count} events")
        
        if errors:
            print(f"\n⚠️  Error details (first 10):")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        
        # Generate summary statistics
        print(f"\n📊 Migration Summary:")
        print(f"  Total records processed: {len(df)}")
        print(f"  Successfully migrated: {migrated_count}")
        print(f"  Success rate: {(migrated_count/len(df)*100):.1f}%")
        
        return migrated_count > 0
        
    except Exception as e:
        logger.error("Migration failed", error=str(e))
        print(f"❌ Migration failed: {str(e)}")
        return False
    
    finally:
        await close_mongo()


async def validate_migrated_data():
    """Validate the migrated data"""
    
    print("\n🔍 Validating migrated data...")
    
    try:
        await init_mongo()
        db = get_db()
        service = HistoricalDataService(db)
        
        # Get basic statistics
        total_events = await db["historical_flood_events"].count_documents({})
        print(f"✅ Total events in database: {total_events}")
        
        # Check data quality
        verified_events = await db["historical_flood_events"].count_documents({"verified": True})
        print(f"📊 Verified events: {verified_events} ({(verified_events/total_events*100):.1f}%)")
        
        # Check severity distribution
        severity_pipeline = [
            {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        severity_dist = await db["historical_flood_events"].aggregate(severity_pipeline).to_list(None)
        
        print(f"\n📈 Severity Distribution:")
        for item in severity_dist:
            print(f"  {item['_id']}: {item['count']} events")
        
        # Check flood type distribution
        type_pipeline = [
            {"$group": {"_id": "$flood_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        type_dist = await db["historical_flood_events"].aggregate(type_pipeline).to_list(None)
        
        print(f"\n🌊 Flood Type Distribution:")
        for item in type_dist:
            print(f"  {item['_id']}: {item['count']} events")
        
        # Check geographic coverage
        locations = await db["historical_flood_events"].distinct("location.name")
        print(f"\n🗺️  Geographic Coverage:")
        print(f"  Unique locations: {len(locations)}")
        
        provinces = await db["historical_flood_events"].distinct("location.province")
        print(f"  Provinces covered: {len([p for p in provinces if p])}")
        
        # Check date range
        earliest = await db["historical_flood_events"].find_one({}, sort=[("start_date", 1)])
        latest = await db["historical_flood_events"].find_one({}, sort=[("start_date", -1)])
        
        if earliest and latest:
            print(f"\n📅 Date Range:")
            print(f"  Earliest event: {earliest['start_date']}")
            print(f"  Latest event: {latest['start_date']}")
        
        print(f"\n✅ Data validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error("Validation failed", error=str(e))
        print(f"❌ Validation failed: {str(e)}")
        return False
    
    finally:
        await close_mongo()


async def create_sample_events():
    """Create some sample events for testing"""
    
    print("\n🧪 Creating sample events for testing...")
    
    try:
        await init_mongo()
        db = get_db()
        service = HistoricalDataService(db)
        
        # Sample events with realistic South African data
        sample_events = [
            {
                "event_id": "SAMPLE_2024_001",
                "name": "Durban Flash Flood 2024",
                "start_date": date(2024, 3, 15),
                "flood_type": FloodType.FLASH_FLOOD,
                "severity": FloodSeverityLevel.SEVERE,
                "location": GeographicLocation(
                    name="Durban",
                    latitude=-29.8587,
                    longitude=31.0218,
                    district="eThekwini",
                    province="KwaZulu-Natal",
                    country="South Africa"
                ),
                "impacts": ImpactMetrics(
                    deaths=12,
                    injuries=45,
                    displaced_persons=500,
                    total_economic_impact_usd=25000000
                ),
                "data_source": "Sample Data",
                "data_quality": "excellent",
                "verified": True
            },
            {
                "event_id": "SAMPLE_2023_002",
                "name": "Cape Town Coastal Storm Surge",
                "start_date": date(2023, 7, 20),
                "flood_type": FloodType.STORM_SURGE,
                "severity": FloodSeverityLevel.MODERATE,
                "location": GeographicLocation(
                    name="Cape Town",
                    latitude=-33.9249,
                    longitude=18.4241,
                    district="City of Cape Town",
                    province="Western Cape",
                    country="South Africa"
                ),
                "impacts": ImpactMetrics(
                    deaths=3,
                    injuries=18,
                    displaced_persons=150,
                    total_economic_impact_usd=8500000
                ),
                "data_source": "Sample Data",
                "data_quality": "excellent",
                "verified": True
            }
        ]
        
        created_count = 0
        for event_data in sample_events:
            try:
                event = HistoricalFloodEvent(**event_data)
                result = await service.create_event(event)
                if result["success"]:
                    created_count += 1
                    print(f"✅ Created sample event: {event_data['name']}")
            except Exception as e:
                print(f"❌ Failed to create sample event: {str(e)}")
        
        print(f"✅ Created {created_count} sample events")
        return created_count > 0
        
    except Exception as e:
        logger.error("Sample creation failed", error=str(e))
        print(f"❌ Sample creation failed: {str(e)}")
        return False
    
    finally:
        await close_mongo()


async def main():
    """Main migration function"""
    
    print("🌟 Crisis Connect - Historical Data Migration")
    print("=" * 50)
    
    # Check if we should run migration
    response = input("\n🔄 Do you want to migrate legacy data? (y/n): ").lower()
    if response in ['y', 'yes']:
        success = await migrate_legacy_data()
        if success:
            await validate_migrated_data()
        else:
            print("❌ Migration failed, skipping validation")
    
    # Check if we should create sample data
    response = input("\n🧪 Do you want to create sample events for testing? (y/n): ").lower()
    if response in ['y', 'yes']:
        await create_sample_events()
    
    print(f"\n🎉 Migration process completed!")
    print(f"📖 Check the API documentation at http://localhost:8000/docs for new endpoints")
    print(f"📊 Use the Streamlit dashboard to explore the migrated data")


if __name__ == "__main__":
    asyncio.run(main())
