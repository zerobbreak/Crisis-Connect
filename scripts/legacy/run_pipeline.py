import asyncio
import structlog
import joblib
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.db import init_mongo, close_mongo, get_db
from services.weather_service import WeatherService
from services.alert_service import AlertService
from services.location_service import LocationService
from services.predict import load_all_baselines

# Configure logging for console output
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()  # Changed from JSONRenderer for better readability
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class MockApp:
    """Mock FastAPI app to hold state for services"""
    def __init__(self):
        self.state = type('State', (), {})()

async def run_pipeline():
    print("=" * 80)
    print(">>> CRISIS CONNECT PIPELINE STARTING")
    print("=" * 80)
    logger.info(">>> Starting Crisis Connect Pipeline")
    
    # 1. Setup Environment
    app = MockApp()
    app.state.executor = ThreadPoolExecutor(max_workers=2)
    
    try:
        print("[DEBUG] Initializing MongoDB...")
        # Initialize DB
        await init_mongo(app)
        db = get_db(app)
        logger.info("[OK] MongoDB initialized")
        print("[DEBUG] MongoDB initialized successfully")
        
        # Load Model
        print("[DEBUG] Loading ML model...")
        try:
            app.state.model = joblib.load(settings.model_path)
            logger.info("[OK] ML Model loaded")
            print("[DEBUG] ML Model loaded successfully")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model: {e}")
            print(f"[DEBUG ERROR] Failed to load model: {e}")
            return

        # Initialize Services
        print("[DEBUG] Initializing services...")
        weather_service = WeatherService(app)
        alert_service = AlertService(app)
        location_service = LocationService(db)
        print("[DEBUG] Services initialized")
        
        # Load Baselines
        logger.info("[WAIT] Loading historical baselines...")
        print("[DEBUG] Loading baselines...")
        load_all_baselines()
        logger.info("[OK] Baselines loaded")
        print("[DEBUG] Baselines loaded successfully")

        # 2. Initialize Locations
        logger.info("[INIT] Initializing default locations...")
        print("[DEBUG] Initializing locations...")
        location_result = await location_service.initialize_default_locations()
        if location_result['success']:
            logger.info(f"[OK] {location_result['message']}")
            print(f"[DEBUG] Locations initialized: {location_result['message']}")
        else:
            logger.error(f"[ERROR] Location initialization failed: {location_result['message']}")
            print(f"[DEBUG ERROR] Location initialization failed: {location_result['message']}")
            return
        
        # 3. Collect Weather Data
        logger.info("[DATA] Collecting weather data...")
        print("[DEBUG] Collecting weather data...")
        collect_result = await weather_service.collect_weather_data()
        if collect_result['success']:
            logger.info(f"[OK] Collected data for {collect_result['count']} locations")
            print(f"[DEBUG] Weather data collected: {collect_result['count']} locations")
        else:
            logger.error(f"[ERROR] Data collection failed: {collect_result['message']}")
            print(f"[DEBUG ERROR] Data collection failed: {collect_result['message']}")
            return
            
        # 4. Assess Risk
        logger.info("[RISK] Assessing risk...")
        print("[DEBUG] Assessing risk...")
        risk_result = await weather_service.process_risk_assessment(app.state.model, generate_alerts=False)
        if risk_result['success']:
            logger.info(f"[OK] Risk assessment complete. Processed: {risk_result['predictions_count']}")
            print(f"[DEBUG] Risk assessment complete: {risk_result['predictions_count']} predictions")
        else:
            logger.error(f"[ERROR] Risk assessment failed: {risk_result['message']}")
            print(f"[DEBUG ERROR] Risk assessment failed: {risk_result['message']}")
            return

        # 5. Generate Alerts
        logger.info("[ALERT] Generating alerts...")
        print("[DEBUG] Generating alerts...")
        alerts = await alert_service.generate_alerts_from_predictions(limit=1000, risk_threshold=70.0)
        logger.info(f"[OK] Generated {len(alerts)} alerts")
        print(f"[DEBUG] Generated {len(alerts)} alerts")
        
        for alert in alerts:
            logger.info("[!] ALERT GENERATED", 
                       location=alert['location'], 
                       risk=alert['risk_level'],
                       message=alert['message'][:100] + "...")
        
        # 6. Display Summary
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        
        # Get predictions from database to show top affected areas
        predictions = await db["predictions"].find(
            {"composite_risk_score": {"$exists": True}},
            {"location": 1, "composite_risk_score": 1, "scenario": 1, "_id": 0}
        ).sort("composite_risk_score", -1).limit(10).to_list(length=10)
        
        print("\n[TOP 10 MOST AFFECTED LOCATIONS]")
        print("-" * 80)
        print(f"{'Rank':<6} {'Location':<35} {'Risk Score':<12} {'Scenario':<15}")
        print("-" * 80)
        
        for idx, pred in enumerate(predictions, 1):
            location = pred.get('location', 'Unknown')
            risk_score = pred.get('composite_risk_score', 0)
            scenario = pred.get('scenario', 'N/A')
            
            # Color code based on risk level
            risk_indicator = "CRITICAL" if risk_score >= 90 else "HIGH" if risk_score >= 70 else "MODERATE"
            
            print(f"{idx:<6} {location:<35} {risk_score:<12.2f} {scenario:<15}")
        
        print("\n[ALERT BREAKDOWN]")
        print("-" * 80)
        
        # Count alerts by severity
        alert_counts = {}
        for alert in alerts:
            severity = alert.get('severity', 'unknown')
            alert_counts[severity] = alert_counts.get(severity, 0) + 1
        
        for severity, count in sorted(alert_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{severity.upper():<15}: {count} alerts")
        
        print("\n[STATISTICS]")
        print("-" * 80)
        print(f"Total Locations Monitored: {collect_result.get('count', 0)}")
        print(f"Risk Assessments Generated: {risk_result.get('predictions_count', 0)}")
        print(f"Total Alerts Generated: {len(alerts)}")
        print(f"High-Risk Locations (>=70): {sum(1 for p in predictions if p.get('composite_risk_score', 0) >= 70)}")
        
        print("\n" + "=" * 80)
        print("[DEBUG] Pipeline completed successfully!")
        logger.info("[SUCCESS] Pipeline completed successfully")

    except Exception as e:
        logger.error("[ERROR] Pipeline failed", error=str(e), exc_info=True)
    finally:
        # Cleanup
        await close_mongo(app)
        app.state.executor.shutdown(wait=False)
        logger.info("[DONE] Pipeline finished")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
