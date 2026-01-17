#!/usr/bin/env python3
"""
Performance Tracker - Track model performance over time
Monitors prediction accuracy, model drift, and improvement trends
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import json
import structlog
from motor.motor_asyncio import AsyncIOMotorClient

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

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


class PerformanceTracker:
    """Track and analyze model performance over time"""
    
    def __init__(self, db_uri: str = None, metrics_file: str = "data/metrics/performance_history.json"):
        self.db_uri = db_uri or "mongodb://localhost:27017"
        self.db_name = "crisisconnect"
        self.metrics_file = metrics_file
        self.client = None
        self.db = None
        
        # Create metrics directory
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.db_uri)
            self.db = self.client[self.db_name]
            await self.client.admin.command('ping')
            logger.info("[OK] Connected to MongoDB")
            return True
        except Exception as e:
            logger.error("[ERROR] Failed to connect to MongoDB", error=str(e))
            return False
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("[OK] MongoDB connection closed")
    
    async def get_prediction_stats(self, days_back: int = 7) -> Dict[str, Any]:
        """Get prediction statistics for the last N days"""
        try:
            start_date = datetime.now() - timedelta(days=days_back)
            
            # Get predictions
            predictions = await self.db["predictions"].find({
                "timestamp": {"$gte": start_date.strftime("%Y-%m-%d %H:%M:%S")}
            }).to_list(length=None)
            
            if not predictions:
                return {
                    "total_predictions": 0,
                    "avg_risk_score": 0,
                    "high_risk_count": 0,
                    "low_risk_count": 0
                }
            
            df = pd.DataFrame(predictions)
            
            stats = {
                "total_predictions": len(df),
                "avg_risk_score": df['composite_risk_score'].mean() if 'composite_risk_score' in df.columns else 0,
                "high_risk_count": len(df[df['composite_risk_score'] >= 70]) if 'composite_risk_score' in df.columns else 0,
                "low_risk_count": len(df[df['composite_risk_score'] < 70]) if 'composite_risk_score' in df.columns else 0,
                "locations": df['location'].nunique() if 'location' in df.columns else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error("[ERROR] Failed to get prediction stats", error=str(e))
            return {}
    
    async def get_alert_stats(self, days_back: int = 7) -> Dict[str, Any]:
        """Get alert statistics for the last N days"""
        try:
            start_date = datetime.now() - timedelta(days=days_back)
            
            # Get alerts
            alerts = await self.db["alerts"].find({
                "timestamp": {"$gte": start_date.strftime("%Y-%m-%d %H:%M:%S")}
            }).to_list(length=None)
            
            if not alerts:
                return {
                    "total_alerts": 0,
                    "by_severity": {}
                }
            
            df = pd.DataFrame(alerts)
            
            # Count by severity
            severity_counts = df['severity'].value_counts().to_dict() if 'severity' in df.columns else {}
            
            stats = {
                "total_alerts": len(df),
                "by_severity": severity_counts,
                "locations": df['location'].nunique() if 'location' in df.columns else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error("[ERROR] Failed to get alert stats", error=str(e))
            return {}
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to file"""
        try:
            # Load existing metrics
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add timestamp
            metrics['timestamp'] = datetime.now().isoformat()
            
            # Append new metrics
            history.append(metrics)
            
            # Keep last 100 entries
            history = history[-100:]
            
            # Save
            with open(self.metrics_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info("[SAVE] Metrics saved", file=self.metrics_file)
            
        except Exception as e:
            logger.error("[ERROR] Failed to save metrics", error=str(e))
    
    async def generate_report(self, days_back: int = 7) -> Dict[str, Any]:
        """Generate performance report"""
        try:
            print("=" * 80)
            print("PERFORMANCE REPORT")
            print("=" * 80)
            print(f"Period: Last {days_back} days")
            print("=" * 80)
            
            # Get statistics
            print("\n[1/2] Collecting prediction statistics...")
            prediction_stats = await self.get_prediction_stats(days_back)
            
            print("[2/2] Collecting alert statistics...")
            alert_stats = await self.get_alert_stats(days_back)
            
            # Display report
            print("\n" + "-" * 80)
            print("PREDICTIONS")
            print("-" * 80)
            print(f"Total Predictions:     {prediction_stats.get('total_predictions', 0)}")
            print(f"Average Risk Score:    {prediction_stats.get('avg_risk_score', 0):.2f}")
            print(f"High Risk (>=70):      {prediction_stats.get('high_risk_count', 0)}")
            print(f"Low Risk (<70):        {prediction_stats.get('low_risk_count', 0)}")
            print(f"Locations Monitored:   {prediction_stats.get('locations', 0)}")
            
            print("\n" + "-" * 80)
            print("ALERTS")
            print("-" * 80)
            print(f"Total Alerts:          {alert_stats.get('total_alerts', 0)}")
            print(f"Affected Locations:    {alert_stats.get('locations', 0)}")
            
            if alert_stats.get('by_severity'):
                print("\nBy Severity:")
                for severity, count in alert_stats['by_severity'].items():
                    print(f"  {severity.upper():<15}: {count}")
            
            print("\n" + "=" * 80)
            
            # Combine metrics
            report = {
                "period_days": days_back,
                "predictions": prediction_stats,
                "alerts": alert_stats,
                "generated_at": datetime.now().isoformat()
            }
            
            # Save metrics
            self.save_metrics(report)
            
            return report
            
        except Exception as e:
            logger.error("[ERROR] Failed to generate report", error=str(e), exc_info=True)
            return {}


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Tracker")
    parser.add_argument("--days", type=int, default=7,
                       help="Number of days to analyze (default: 7)")
    
    args = parser.parse_args()
    
    tracker = PerformanceTracker()
    
    try:
        # Connect to database
        if not await tracker.connect():
            print("[ERROR] Failed to connect to database")
            return
        
        # Generate report
        report = await tracker.generate_report(days_back=args.days)
        
        if report:
            print(f"\n[SUCCESS] Report generated and saved to: {tracker.metrics_file}")
        else:
            print("\n[FAILED] Report generation failed")
            
    finally:
        await tracker.close()


if __name__ == "__main__":
    asyncio.run(main())
