#!/usr/bin/env python3
"""
Crisis Connect - Unified System
Combines pipeline execution, continuous monitoring, and data aggregation
"""
import asyncio
import signal
import sys
import os
import argparse
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import structlog
import joblib
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from utils.db import init_mongo, close_mongo, get_db
from services.weather_service import WeatherService
from services.alert_service import AlertService
from services.location_service import LocationService
from motor.motor_asyncio import AsyncIOMotorClient

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
    print("\n[SHUTDOWN] Received shutdown signal. Finishing current operation...")
    logger.info("[SHUTDOWN] Graceful shutdown initiated")
    shutdown_requested = True


class MockApp:
    """Mock FastAPI app to hold state for services"""
    def __init__(self):
        self.state = type('State', (), {})()


class CrisisConnectSystem:
    """Unified Crisis Connect system combining all components"""
    
    def __init__(self, 
                 mode: str = "once",
                 interval: int = 300,
                 risk_threshold: float = 70.0,
                 max_alerts: int = 1000,
                 aggregate_days: int = 7):
        self.mode = mode  # "once", "watch", "aggregate"
        self.interval = interval
        self.risk_threshold = risk_threshold
        self.max_alerts = max_alerts
        self.aggregate_days = aggregate_days
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
        print(">>> CRISIS CONNECT SYSTEM STARTING")
        print("=" * 80)
        print(f"Mode: {self.mode.upper()}")
        if self.mode == "watch":
            print(f"Monitor Interval: {self.interval} seconds ({self.interval/60:.1f} minutes)")
            print(f"Risk Threshold: {self.risk_threshold}")
            print(f"Max Alerts: {self.max_alerts}")
        elif self.mode == "aggregate":
            print(f"Aggregation Period: Last {self.aggregate_days} days")
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
            
            if current_modified > self.model_last_modified:
                logger.info("[MODEL] New model version detected, reloading...")
                print(f"[MODEL] Reloading updated model...")
                
                self.app.state.model = joblib.load(self.current_model_path)
                self.model_last_modified = current_modified
                
                logger.info("[MODEL] Model reloaded successfully")
                print(f"[MODEL] Model updated successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error("[MODEL] Failed to reload model", error=str(e))
            return False
    
    async def run_pipeline_once(self):
        """Execute the pipeline once"""
        print("\n[PIPELINE] Starting single execution...")
        
        try:
            # 1. Initialize locations
            print("[1/4] Initializing locations...")
            location_result = await self.location_service.initialize_default_locations()
            if location_result['success']:
                print(f"[OK] {location_result['message']}")
            else:
                print(f"[ERROR] {location_result['message']}")
                return False
            
            # 2. Collect weather data
            print("[2/4] Collecting weather data...")
            collect_result = await self.weather_service.collect_weather_data()
            if not collect_result['success']:
                print(f"[ERROR] {collect_result['message']}")
                return False
            print(f"[OK] Collected data for {collect_result['count']} locations")
            
            # 3. Run risk assessment
            print("[3/4] Running risk assessment...")
            risk_result = await self.weather_service.process_risk_assessment(
                self.app.state.model,
                generate_alerts=False
            )
            if not risk_result['success']:
                print(f"[ERROR] {risk_result['message']}")
                return False
            print(f"[OK] Generated {risk_result['predictions_count']} predictions")
            
            # 4. Generate alerts
            print("[4/4] Generating alerts...")
            alerts = await self.alert_service.generate_alerts_from_predictions(
                limit=self.max_alerts,
                risk_threshold=self.risk_threshold
            )
            print(f"[OK] Generated {len(alerts)} alerts")
            
            # Display summary
            await self.display_summary(collect_result, risk_result, alerts)
            
            return True
            
        except Exception as e:
            logger.error("[PIPELINE] Execution failed", error=str(e), exc_info=True)
            print(f"[ERROR] Pipeline failed: {str(e)}")
            return False
    
    async def run_monitoring_cycle(self):
        """Execute one monitoring cycle"""
        self.cycle_count += 1
        cycle_start = time.time()
        
        print(f"\n[CYCLE {self.cycle_count}] Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Check for model updates
            await self.check_and_reload_model()
            
            # Collect weather data
            print(f"[CYCLE {self.cycle_count}] Collecting weather data...")
            collect_result = await self.weather_service.collect_weather_data()
            
            if not collect_result['success']:
                logger.error(f"[CYCLE {self.cycle_count}] Data collection failed")
                return False
            
            locations_count = collect_result['count']
            print(f"[CYCLE {self.cycle_count}] Collected data for {locations_count} locations")
            
            # Run risk assessment
            print(f"[CYCLE {self.cycle_count}] Running risk assessment...")
            risk_result = await self.weather_service.process_risk_assessment(
                self.app.state.model,
                generate_alerts=False
            )
            
            if not risk_result['success']:
                logger.error(f"[CYCLE {self.cycle_count}] Risk assessment failed")
                return False
            
            predictions_count = risk_result['predictions_count']
            print(f"[CYCLE {self.cycle_count}] Generated {predictions_count} predictions")
            
            # Generate alerts
            print(f"[CYCLE {self.cycle_count}] Generating alerts...")
            alerts = await self.alert_service.generate_alerts_from_predictions(
                limit=self.max_alerts,
                risk_threshold=self.risk_threshold
            )
            
            alerts_count = len(alerts)
            print(f"[CYCLE {self.cycle_count}] Generated {alerts_count} alerts")
            
            # Get high-risk locations
            high_risk_locations = await self.db["predictions"].count_documents({
                "composite_risk_score": {"$gte": self.risk_threshold}
            })
            
            # Log summary
            cycle_duration = time.time() - cycle_start
            print(f"[CYCLE {self.cycle_count}] Completed in {cycle_duration:.1f}s")
            print(f"[CYCLE {self.cycle_count}] Summary: {locations_count} locations, "
                  f"{predictions_count} predictions, {alerts_count} alerts, "
                  f"{high_risk_locations} high-risk")
            
            return True
            
        except Exception as e:
            logger.error(f"[CYCLE {self.cycle_count}] Failed", error=str(e), exc_info=True)
            return False
    
    async def run_data_aggregation(self):
        """Aggregate data for model retraining"""
        print("\n[AGGREGATION] Starting data collection...")
        
        try:
            start_date = datetime.now() - timedelta(days=self.aggregate_days)
            end_date = datetime.now()
            
            # Fetch weather data
            print("[1/3] Fetching weather data...")
            weather_cursor = self.db["weather_data"].find({
                "timestamp": {
                    "$gte": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "$lte": end_date.strftime("%Y-%m-%d %H:%M:%S")
                }
            })
            weather_records = await weather_cursor.to_list(length=None)
            weather_df = pd.DataFrame(weather_records) if weather_records else pd.DataFrame()
            print(f"[OK] Found {len(weather_df)} weather records")
            
            # Fetch predictions
            print("[2/3] Fetching predictions...")
            pred_cursor = self.db["predictions"].find({
                "timestamp": {
                    "$gte": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "$lte": end_date.strftime("%Y-%m-%d %H:%M:%S")
                }
            })
            pred_records = await pred_cursor.to_list(length=None)
            pred_df = pd.DataFrame(pred_records) if pred_records else pd.DataFrame()
            print(f"[OK] Found {len(pred_df)} prediction records")
            
            # Combine datasets
            print("[3/3] Combining and cleaning data...")
            if not weather_df.empty and not pred_df.empty:
                combined_df = pd.merge(
                    weather_df,
                    pred_df,
                    on=['location', 'timestamp'],
                    how='outer',
                    suffixes=('_weather', '_pred')
                )
            elif not weather_df.empty:
                combined_df = weather_df
            elif not pred_df.empty:
                combined_df = pred_df
            else:
                print("[WARNING] No data available for aggregation")
                return False
            
            # Clean data
            # Remove MongoDB _id fields if present
            cols_to_drop = [c for c in combined_df.columns if '_id' in c]
            if cols_to_drop:
                combined_df = combined_df.drop(columns=cols_to_drop)
            
            initial_count = len(combined_df)
            # Use subset to avoid unhashable type error with dict fields
            combined_df = combined_df.drop_duplicates(subset=['location', 'timestamp'])
            combined_df = combined_df.dropna(subset=['location'], how='any')
            
            print(f"[OK] Cleaned data: {len(combined_df)}/{initial_count} records retained")
            
            # Save to file
            output_file = "data/training/aggregated_data.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            combined_df.to_csv(output_file, index=False)
            
            print(f"\n[SUCCESS] Aggregated {len(combined_df)} records")
            print(f"[SAVED] Data exported to: {output_file}")
            print(f"[INFO] Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            return True
            
        except Exception as e:
            logger.error("[AGGREGATION] Failed", error=str(e), exc_info=True)
            print(f"[ERROR] Aggregation failed: {str(e)}")
            return False
    
    async def display_summary(self, collect_result, risk_result, alerts):
        """Display pipeline summary"""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        # Get top affected locations
        predictions = await self.db["predictions"].find(
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
            print(f"{idx:<6} {location:<35} {risk_score:<12.2f} {scenario:<15}")
        
        print("\n[ALERT BREAKDOWN]")
        print("-" * 80)
        
        # Count alerts by severity
        alert_counts = {}
        for alert in alerts:
            severity = alert.get('severity', alert.get('risk_level', 'unknown'))
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
    
    async def run_watch_mode(self):
        """Run continuous monitoring"""
        global shutdown_requested
        
        print(f"\n[WATCHER] Monitoring started. Press CTRL+C to stop.")
        print(f"[WATCHER] First cycle starting now...\n")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while not shutdown_requested:
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
                
                if shutdown_requested:
                    break
                
                # Sleep until next cycle
                next_cycle_time = datetime.now().timestamp() + self.interval
                print(f"[WATCHER] Next cycle in {self.interval}s at {datetime.fromtimestamp(next_cycle_time).strftime('%H:%M:%S')}")
                
                sleep_remaining = self.interval
                while sleep_remaining > 0 and not shutdown_requested:
                    sleep_time = min(1, sleep_remaining)
                    await asyncio.sleep(sleep_time)
                    sleep_remaining -= sleep_time
                    
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Keyboard interrupt received")
    
    async def run(self):
        """Main execution based on mode"""
        # Initialize
        if not await self.initialize():
            print("[ERROR] Failed to initialize system. Exiting.")
            return False
        
        try:
            if self.mode == "once":
                return await self.run_pipeline_once()
            elif self.mode == "watch":
                await self.run_watch_mode()
                return True
            elif self.mode == "aggregate":
                return await self.run_data_aggregation()
            else:
                print(f"[ERROR] Unknown mode: {self.mode}")
                return False
                
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
        
        if self.mode == "watch":
            print(f"[SHUTDOWN] Watcher stopped after {self.cycle_count} cycles")
        
        logger.info("[SHUTDOWN] Complete")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Crisis Connect - Unified System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline once
  python crisis_connect.py

  # Continuous monitoring (5 minutes)
  python crisis_connect.py --watch

  # Continuous monitoring (custom interval)
  python crisis_connect.py --watch --interval 60

  # Aggregate data for retraining
  python crisis_connect.py --aggregate --days 7
        """
    )
    
    parser.add_argument("--watch", action="store_true",
                       help="Run in continuous monitoring mode")
    parser.add_argument("--aggregate", action="store_true",
                       help="Aggregate data for model retraining")
    parser.add_argument("--interval", type=int, default=300,
                       help="Monitoring interval in seconds (default: 300)")
    parser.add_argument("--threshold", type=float, default=70.0,
                       help="Risk threshold for alerts (default: 70.0)")
    parser.add_argument("--max-alerts", type=int, default=1000,
                       help="Maximum alerts per cycle (default: 1000)")
    parser.add_argument("--days", type=int, default=7,
                       help="Days to aggregate (default: 7)")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.watch:
        mode = "watch"
    elif args.aggregate:
        mode = "aggregate"
    else:
        mode = "once"
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run system
    system = CrisisConnectSystem(
        mode=mode,
        interval=args.interval,
        risk_threshold=args.threshold,
        max_alerts=args.max_alerts,
        aggregate_days=args.days
    )
    
    success = await system.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
