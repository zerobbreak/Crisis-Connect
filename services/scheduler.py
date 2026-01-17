"""
Background Scheduler Service
Manages periodic tasks for data updates, forecasting, and alerts.
"""

import logging
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Services
from services.flood_data import flood_service
from services.time_series_data import create_time_series_service
from services.forecast import create_forecast_service
from services.telegram_bot import telegram_service
from services.alert_engine import alert_engine
from services.data_ingestion import DataIngestionService

logger = logging.getLogger("crisisconnect.scheduler")


class SchedulerService:
    """Service to manage background jobs for Phase 3.

    Jobs:
    * update_live_data – fetch weather & flood data for active locations
    * run_forecasts – generate forecasts for locations
    * process_alerts – evaluate forecasts and send alerts
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        # In a real app this would be populated from the DB
        self.demo_location = {"_id": "demo", "name": "Demo Location", "latitude": -29.8587, "longitude": 31.0218}

    def start(self):
        """Start the APScheduler instance and schedule jobs."""
        if not self.is_running:
            self._add_jobs()
            self.scheduler.start()
            self.is_running = True
            logger.info("Background scheduler started")

    def shutdown(self):
        """Shutdown the scheduler without waiting for running jobs (avoids CancelledError)."""
        if self.is_running:
            self.scheduler.shutdown(wait=False)
            self.is_running = False
            logger.info("Background scheduler stopped")

    def _add_jobs(self):
        """Define and add periodic jobs to the scheduler."""
        # Job 1: Update Live Data (hourly)
        self.scheduler.add_job(
            self.update_live_data,
            IntervalTrigger(hours=1),
            id="update_live_data",
            replace_existing=True,
            next_run_time=datetime.now(),  # run immediately on startup
        )
        # Job 2: Run Forecasts (every 6 hours)
        self.scheduler.add_job(
            self.run_forecasts,
            IntervalTrigger(hours=6),
            id="run_forecasts",
            replace_existing=True,
        )
        # Job 3: Process Alerts (every 15 minutes)
        self.scheduler.add_job(
            self.process_alerts,
            IntervalTrigger(minutes=15),
            id="process_alerts",
            replace_existing=True,
        )

    async def update_live_data(self):
        """Fetch latest weather and flood data for active locations."""
        logger.info("Job: Updating live data…")
        try:
            ingestion = DataIngestionService()
            # For demo we use a single location; replace with DB query in production
            data = await ingestion.fetch_for_location(self.demo_location)
            logger.info(
                "Live data fetched",
                location_id=self.demo_location["_id"],
                data=data,
            )
        except asyncio.CancelledError:
            logger.info("update_live_data job cancelled during shutdown")
        except Exception as e:
            logger.error(f"Job failed: update_live_data: {e}")

    async def run_forecasts(self):
        """Generate forecasts for active locations and persist them."""
        logger.info("Job: Running periodic forecasts…")
        try:
            # Demo: generate forecast for the demo location
            ts_service = await create_time_series_service(
                self.demo_location["latitude"], self.demo_location["longitude"]
            )
            forecast_service = await create_forecast_service(ts_service)
            forecast = await forecast_service.predict_horizon(6)
            logger.info(
                "Forecast generated",
                location_id=self.demo_location["_id"],
                forecast=forecast,
            )
            # In a full implementation we would persist the forecast to storage here.
        except asyncio.CancelledError:
            logger.info("run_forecasts job cancelled during shutdown")
        except Exception as e:
            logger.error(f"Job failed: run_forecasts: {e}")

    async def process_alerts(self):
        """Evaluate forecasts against thresholds and send alerts if needed."""
        logger.info("Job: Processing alerts…")
        try:
            # Demo: reuse the same forecast generation logic
            ts_service = await create_time_series_service(
                self.demo_location["latitude"], self.demo_location["longitude"]
            )
            forecast_service = await create_forecast_service(ts_service)
            forecast = await forecast_service.predict_horizon(6)
            # Evaluate risk using the AlertEngine
            alert = await alert_engine.evaluate_risk(self.demo_location, {"forecasts": forecast.get("forecasts", [])})
            if alert:
                await alert_engine.process_alert(alert)
        except asyncio.CancelledError:
            logger.info("process_alerts job cancelled during shutdown")
        except Exception as e:
            logger.error(f"Job failed: process_alerts: {e}")


# Singleton instance used by the application
scheduler_service = SchedulerService()
