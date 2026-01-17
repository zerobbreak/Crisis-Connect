#!/usr/bin/env python3
"""
Weather Watcher - Continuous Weather Monitoring Service
Monitors weather conditions, runs risk assessments, and generates alerts automatically.
"""
import asyncio
import signal
import sys
import os
import argparse
import time
from datetime import datetime
from typing import Optional
import structlog
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.db import init_mongo, close_mongo, get_db
from services.weather_service import WeatherService
from services.alert_service import AlertService
from services.location_service import LocationService

# Configure logging
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
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    print("\n[SHUTDOWN] Received shutdown signal. Finishing current cycle...")
    logger.info("[SHUTDOWN] Graceful shutdown initiated")
    shutdown_requested = True


class MockApp:
    """Mock FastAPI app to hold state for services"""
    def __init__(self):
        self.state = type('State', (), {})()


class WeatherWatcher:
    """Continuous weather monitoring service"""
    
    def __init__(self, interval: int = 300, risk_threshold: float = 70.0, max_alerts: int = 1000):
        self.interval = interval
        self.risk_threshold = risk_threshold
        self.max_alerts = max_alerts
        self.cycle_count = 0
        self.app = None
        self.weather_service = None
        self.alert_service = None
        self.location_service = None
        self.db = None
        self.current_model_path = settings.model_path
        self.model_last_modified = None
        
    async def initialize(self):
        """Initialize services and database"""
        print("=" * 80)
        print(">>> WEATHER WATCHER STARTING")
        print("=" * 80)
        print(f"Configuration:")
        print(f"  - Monitor Interval: {self.interval} seconds ({self.interval/60:.1f} minutes)")
        print(f"  - Risk Threshold: {self.risk_threshold}")
        print(f"  - Max Alerts: {self.max_alerts}")
        print("=" * 80)
        
        # Setup app
        self.app = MockApp()
        
        try:
            # Initialize database
            await init_mongo(self.app)
            self.db = get_db(self.app)
            logger.info("[OK] MongoDB connected")
            
            # Load model
            self.app.state.model = joblib.load(settings.model_path)
            self.model_last_modified = os.path.getmtime(settings.model_path)
            logger.info("[OK] ML Model loaded", path=settings.model_path)
            
            # Initialize services
            self.weather_service = WeatherService(self.app)
            self.alert_service = AlertService(self.app)
            self.location_service = LocationService(self.db)
            logger.info("[OK] Services initialized")
            
            return True
            
        except Exception as e:
            logger.error("[ERROR] Initialization failed", error=str(e), exc_info=True)
            return False
    
    async def check_and_reload_model(self):
        """Check if model has been updated and reload if necessary"""
        try:
            if not os.path.exists(self.current_model_path):
                return False
            
            current_modified = os.path.getmtime(self.current_model_path)
            
            # Check if model file has been modified
            if current_modified > self.model_last_modified:
                logger.info("[MODEL] New model version detected, reloading...",
                           old_time=datetime.fromtimestamp(self.model_last_modified),
                           new_time=datetime.fromtimestamp(current_modified))
                print(f"[MODEL] Reloading updated model...")
                
                # Reload model
                self.app.state.model = joblib.load(self.current_model_path)
                self.model_last_modified = current_modified
                
                logger.info("[MODEL] Model reloaded successfully")
                print(f"[MODEL] Model updated successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error("[MODEL] Failed to reload model", error=str(e))
            return False
    
    async def run_monitoring_cycle(self):
        """Execute one monitoring cycle"""
        self.cycle_count += 1
        cycle_start = time.time()
        
        print(f"\n[CYCLE {self.cycle_count}] Starting monitoring cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"[CYCLE {self.cycle_count}] Starting")
        
        try:
            # 0. Check for model updates
            await self.check_and_reload_model()
            
            # 1. Collect Weather Data
            print(f"[CYCLE {self.cycle_count}] Collecting weather data...")
            collect_result = await self.weather_service.collect_weather_data()
            
            if not collect_result['success']:
                logger.error(f"[CYCLE {self.cycle_count}] Data collection failed", 
                           message=collect_result['message'])
                return False
            
            locations_count = collect_result['count']
            print(f"[CYCLE {self.cycle_count}] Collected data for {locations_count} locations")
            
            # 2. Run Risk Assessment
            print(f"[CYCLE {self.cycle_count}] Running risk assessment...")
            risk_result = await self.weather_service.process_risk_assessment(
                self.app.state.model, 
                generate_alerts=False
            )
            
            if not risk_result['success']:
                logger.error(f"[CYCLE {self.cycle_count}] Risk assessment failed",
                           message=risk_result['message'])
                return False
            
            predictions_count = risk_result['predictions_count']
            print(f"[CYCLE {self.cycle_count}] Generated {predictions_count} risk predictions")
            
            # 3. Generate Alerts
            print(f"[CYCLE {self.cycle_count}] Generating alerts...")
            alerts = await self.alert_service.generate_alerts_from_predictions(
                limit=self.max_alerts,
                risk_threshold=self.risk_threshold
            )
            
            alerts_count = len(alerts)
            print(f"[CYCLE {self.cycle_count}] Generated {alerts_count} alerts")
            
            # 4. Get high-risk locations
            high_risk_locations = await self.db["predictions"].count_documents({
                "composite_risk_score": {"$gte": self.risk_threshold}
            })
            
            # Log summary
            cycle_duration = time.time() - cycle_start
            print(f"[CYCLE {self.cycle_count}] Completed in {cycle_duration:.1f}s")
            print(f"[CYCLE {self.cycle_count}] Summary: {locations_count} locations, "
                  f"{predictions_count} predictions, {alerts_count} alerts, "
                  f"{high_risk_locations} high-risk")
            
            logger.info(f"[CYCLE {self.cycle_count}] Completed",
                       duration=cycle_duration,
                       locations=locations_count,
                       predictions=predictions_count,
                       alerts=alerts_count,
                       high_risk=high_risk_locations)
            
            return True
            
        except Exception as e:
            logger.error(f"[CYCLE {self.cycle_count}] Failed", error=str(e), exc_info=True)
            print(f"[CYCLE {self.cycle_count}] ERROR: {str(e)}")
            return False
    
    async def run(self):
        """Main monitoring loop"""
        global shutdown_requested
        
        # Initialize
        if not await self.initialize():
            print("[ERROR] Failed to initialize watcher. Exiting.")
            return
        
        print(f"\n[WATCHER] Monitoring started. Press CTRL+C to stop.")
        print(f"[WATCHER] First cycle starting now...\n")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while not shutdown_requested:
                # Run monitoring cycle
                success = await self.run_monitoring_cycle()
                
                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning(f"[WATCHER] Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("[WATCHER] Too many consecutive failures. Stopping.")
                        print(f"\n[ERROR] {max_consecutive_failures} consecutive failures. Stopping watcher.")
                        break
                
                # Check for shutdown before sleeping
                if shutdown_requested:
                    break
                
                # Sleep until next cycle
                next_cycle_time = datetime.now().timestamp() + self.interval
                print(f"[WATCHER] Next cycle in {self.interval}s at {datetime.fromtimestamp(next_cycle_time).strftime('%H:%M:%S')}")
                
                # Sleep in small increments to allow quick shutdown
                sleep_remaining = self.interval
                while sleep_remaining > 0 and not shutdown_requested:
                    sleep_time = min(1, sleep_remaining)  # Sleep 1 second at a time
                    await asyncio.sleep(sleep_time)
                    sleep_remaining -= sleep_time
                    
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Keyboard interrupt received")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        print("\n[SHUTDOWN] Cleaning up...")
        logger.info("[SHUTDOWN] Cleanup started")
        
        try:
            if self.app:
                await close_mongo(self.app)
                logger.info("[SHUTDOWN] Database closed")
        except Exception as e:
            logger.error("[SHUTDOWN] Cleanup error", error=str(e))
        
        print(f"[SHUTDOWN] Watcher stopped after {self.cycle_count} cycles")
        logger.info(f"[SHUTDOWN] Complete. Total cycles: {self.cycle_count}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Weather Watcher - Continuous Monitoring Service")
    parser.add_argument("--interval", type=int, default=300,
                       help="Monitoring interval in seconds (default: 300 = 5 minutes)")
    parser.add_argument("--threshold", type=float, default=70.0,
                       help="Risk threshold for alerts (default: 70.0)")
    parser.add_argument("--max-alerts", type=int, default=1000,
                       help="Maximum alerts per cycle (default: 1000)")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run watcher
    watcher = WeatherWatcher(
        interval=args.interval,
        risk_threshold=args.threshold,
        max_alerts=args.max_alerts
    )
    
    await watcher.run()


if __name__ == "__main__":
    asyncio.run(main())
