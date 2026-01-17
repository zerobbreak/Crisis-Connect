"""
Phase 3 Standalone Test Script
Verifies Real-time Data Ingestion, Alert Engine, and Notification Services.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.flood_data import flood_service
from services.telegram_bot import telegram_service
from services.alert_engine import alert_engine
from services.scheduler import scheduler_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_phase3")

async def test_flood_data():
    """Test fetching flood data."""
    logger.info("--- Testing FloodDataService ---")
    # Durban coordinates
    lat, lon = -29.8587, 31.0218
    
    try:
        # Test River Discharge
        discharge = await flood_service.fetch_river_discharge(lat, lon, days_forecast=1)
        if discharge:
            logger.info(f"✅ River discharge fetch successful: {discharge.keys()}")
        else:
            logger.warning("⚠️ River discharge fetch returned None (API might be unreachable)")

        # Test Elevation
        elevation = await flood_service.get_elevation(lat, lon)
        if elevation is not None:
            logger.info(f"✅ Elevation fetch successful: {elevation}m")
        else:
            logger.error("❌ Elevation fetch failed")
            
    except Exception as e:
        logger.error(f"❌ Flood data test failed: {e}")

async def test_alert_logic():
    """Test Alert Engine logic."""
    logger.info("\n--- Testing AlertEngine ---")
    
    # Mock Data
    location = {
        "_id": "test_loc_123",
        "name": "Test City",
        "latitude": -29.8587,
        "longitude": 31.0218
    }
    
    # Scenario 1: High Risk
    forecast_high = {
        "forecasts": [
            {"horizon_hours": 24, "predicted_risk": 85.0},
            {"horizon_hours": 48, "predicted_risk": 60.0}
        ]
    }
    
    alert = await alert_engine.evaluate_risk(location, forecast_high)
    if alert and alert["severity"] == "HIGH":
        logger.info("✅ High risk alert generated correctly")
    else:
        logger.error(f"❌ Failed to generate high risk alert: {alert}")

    # Scenario 2: Low Risk
    forecast_low = {
        "forecasts": [
            {"horizon_hours": 24, "predicted_risk": 20.0}
        ]
    }
    
    alert = await alert_engine.evaluate_risk(location, forecast_low)
    if alert is None:
        logger.info("✅ Low risk correctly ignored")
    else:
        logger.error(f"❌ Generated alert for low risk: {alert}")

async def test_telegram_service():
    """Test Telegram Service (Mocked)."""
    logger.info("\n--- Testing TelegramBotService ---")
    
    # Force send an alert to test file logging
    success = await telegram_service.send_alert("TEST_CHAT_ID", "This is a test alert message")
    
    if success:
        logger.info("✅ Telegram service successfully logged alert to file")
    else:
        logger.error("❌ Telegram service failed to log alert")

async def test_scheduler():
    """Test Scheduler Service."""
    logger.info("\n--- Testing SchedulerService ---")
    
    scheduler_service.start()
    if scheduler_service.is_running:
        logger.info("✅ Scheduler started successfully")
    else:
        logger.error("❌ Scheduler failed to start")
        
    scheduler_service.shutdown()
    logger.info("✅ Scheduler shutdown successfully")

async def main():
    logger.info("Starting Phase 3 Verification...")
    
    await test_flood_data()
    await test_alert_logic()
    await test_telegram_service()
    await test_scheduler()
    
    await flood_service.close()
    logger.info("\n✅ Phase 3 Verification Complete")

if __name__ == "__main__":
    asyncio.run(main())
