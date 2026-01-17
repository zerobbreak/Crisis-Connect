"""
Data Collection Scheduler

Automated daily data collection pipeline using APScheduler.
Runs at 2 AM to:
1. Fetch new EM-DAT disasters
2. Fetch weather data for key locations
3. Run cleaning pipeline
4. Engineer features
5. Check quality
6. Update master dataset
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.data_sources.emdat_fetcher import EMDATFetcher
from services.data_sources.emdat_file_loader import EMDATFileLoader
from services.data_sources.noaa_fetcher import NOAAFetcher
from services.data_storage import DataStorage
from services.data_cleaning import DataCleaner
from services.feature_engineering import FeatureEngineer
from services.data_quality_monitor import DataQualityMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/collection.log"),
    ]
)
logger = logging.getLogger(__name__)


class DataCollectionScheduler:
    """Automated data collection and processing pipeline"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize scheduler with pipeline components
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.storage = DataStorage(str(self.data_dir))
        self.emdat = EMDATFetcher()
        self.noaa = NOAAFetcher()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.monitor = DataQualityMonitor()
        
        # Scheduler
        self.scheduler = AsyncIOScheduler()
    
    def start(self, hour: int = 2, minute: int = 0):
        """
        Start the scheduled data collection
        
        Args:
            hour: Hour to run (24-hour format)
            minute: Minute to run
        """
        # Schedule daily job
        self.scheduler.add_job(
            self.collect_and_process,
            trigger=CronTrigger(hour=hour, minute=minute),
            id="daily_collection",
            name="Daily Data Collection",
            replace_existing=True,
        )
        
        self.scheduler.start()
        logger.info(f"Scheduler started. Daily collection at {hour:02d}:{minute:02d}")
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")
    
    async def collect_and_process(self):
        """
        Main data collection and processing pipeline
        
        Steps:
        1. Fetch new EM-DAT disasters
        2. Fetch weather data for key locations
        3. Consolidate all data
        4. Clean the data
        5. Engineer features
        6. Check quality
        7. Save master dataset
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Starting daily data collection...")
        logger.info("=" * 60)
        
        try:
            # Step 1: Fetch EM-DAT data
            logger.info("Step 1/6: Fetching EM-DAT disasters...")
            emdat_df = None
            
            # Try API first (if API key is configured)
            api_key = os.getenv("EMDAT_API_KEY")
            if api_key:
                try:
                    logger.info("  Attempting EM-DAT API fetch...")
                    async with EMDATFetcher() as emdat:
                        emdat_df = await emdat.fetch_disasters(
                            start_year=2010,
                            end_year=datetime.now().year,
                        )
                    
                    if len(emdat_df) > 0:
                        self.storage.save_raw(emdat_df, "emdat")
                        logger.info(f"  [OK] Fetched {len(emdat_df)} EM-DAT records via API")
                    else:
                        logger.warning("  [WARN] No EM-DAT records from API")
                except Exception as e:
                    logger.warning(f"  [WARN] EM-DAT API failed: {e}")
                    logger.info("  Falling back to file loader...")
            else:
                logger.info("  No EM-DAT API key configured, trying file loader...")
            
            # Fallback to file loader if API failed or not configured
            if emdat_df is None or len(emdat_df) == 0:
                try:
                    file_loader = EMDATFileLoader(str(self.data_dir))
                    
                    # Check for downloaded EM-DAT files in data directory
                    emdat_files = list(self.data_dir.glob("*.xlsx")) + list(self.data_dir.glob("*.xls"))
                    emdat_files = [f for f in emdat_files if "emdat" in f.name.lower() or "disaster" in f.name.lower()]
                    
                    if emdat_files:
                        # Use most recent file
                        latest_file = sorted(emdat_files, key=lambda x: x.stat().st_mtime)[-1]
                        logger.info(f"  Loading EM-DAT from file: {latest_file.name}")
                        # Load all regions from 2009+ to meet 1000+ record requirement
                        emdat_df = file_loader.load_excel(
                            str(latest_file),
                            filter_region=False,  # Include all regions, not just Southern Africa
                            start_year=2009,  # Slightly earlier to ensure 1000+ records
                        )
                        
                        if len(emdat_df) > 0:
                            self.storage.save_raw(emdat_df, "emdat_file")
                            logger.info(f"  [OK] Loaded {len(emdat_df)} EM-DAT records from file")
                        else:
                            logger.warning("  [WARN] No records loaded from EM-DAT file")
                    else:
                        logger.warning("  [WARN] No EM-DAT files found in data directory")
                        logger.info("  → To use EM-DAT file loader:")
                        logger.info("    1. Download data from https://www.emdat.be/")
                        logger.info("    2. Save Excel/JSON file to data/ directory")
                        logger.info("    3. Re-run collection")
                except Exception as e:
                    logger.error(f"  [FAIL] EM-DAT file loader failed: {e}")
                    emdat_df = None
            
            # Step 2: Fetch weather data
            logger.info("Step 2/6: Fetching weather data...")
            try:
                async with NOAAFetcher() as noaa:
                    weather_df = await noaa.fetch_multiple_locations(
                        start_date="2020-01-01",  # Last 5 years for daily updates
                        end_date=datetime.now().strftime("%Y-%m-%d"),
                    )
                
                if len(weather_df) > 0:
                    self.storage.save_raw(weather_df, "weather")
                    logger.info(f"  [OK] Fetched {len(weather_df)} weather records")
                else:
                    logger.warning("  [WARN] No weather records fetched")
            except Exception as e:
                logger.error(f"  [FAIL] Weather fetch failed: {e}")
                weather_df = None
            
            # Step 3: Consolidate data
            logger.info("Step 3/6: Consolidating data sources...")
            disasters_df = self.storage.consolidate_disasters(
                sources=["emdat"],
                include_local=True,
            )
            logger.info(f"  [OK] Consolidated {len(disasters_df)} disaster records")
            
            if len(disasters_df) == 0:
                logger.error("  [FAIL] No data to process after consolidation")
                return
            
            # Step 4: Clean data
            logger.info("Step 4/6: Cleaning data...")
            cleaned_df = self.cleaner.clean_disasters(disasters_df)
            report = self.cleaner.get_cleaning_report()
            logger.info(f"  [OK] After cleaning: {len(cleaned_df)} records")
            logger.info(f"    Removed: {report.get('exact_duplicates', 0)} exact dupes, "
                       f"{report.get('near_duplicates', 0)} near dupes")
            
            # Step 5: Engineer features
            logger.info("Step 5/6: Engineering features...")
            featured_df = self.engineer.engineer_disaster_features(cleaned_df)
            logger.info(f"  [OK] Created {len(featured_df.columns)} total features")
            
            # Step 6: Quality check
            logger.info("Step 6/6: Assessing data quality...")
            quality_report = self.monitor.assess_quality(featured_df, source="pipeline")
            self.monitor.save_report(quality_report)
            logger.info(f"  [OK] Quality score: {quality_report['overall_score']:.1f}/100")
            
            if quality_report.get("recommendations"):
                for rec in quality_report["recommendations"][:3]:
                    logger.info(f"    → {rec}")
            
            # Save master dataset
            self.storage.save_processed(featured_df, "disasters_master")
            logger.info("  [OK] Master dataset saved")
            
            # Cleanup old versions
            self.storage.cleanup_old_versions(keep_last=5)
            
            # Summary
            duration = (datetime.now() - start_time).total_seconds()
            logger.info("=" * 60)
            logger.info(f"[SUCCESS] Collection complete in {duration:.1f}s")
            logger.info(f"   Records: {len(featured_df)}")
            logger.info(f"   Features: {len(featured_df.columns)}")
            logger.info(f"   Quality: {quality_report['overall_score']:.1f}/100")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise
    
    async def run_once(self):
        """Run the pipeline once (for testing or manual execution)"""
        await self.collect_and_process()


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Crisis Connect Data Collection Scheduler")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run collection once and exit (for testing)",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=2,
        help="Hour to run daily collection (24-hour format, default: 2)",
    )
    parser.add_argument(
        "--minute",
        type=int,
        default=0,
        help="Minute to run daily collection (default: 0)",
    )
    args = parser.parse_args()
    
    scheduler = DataCollectionScheduler()
    
    if args.once:
        logger.info("Running single collection...")
        await scheduler.run_once()
    else:
        # Start scheduled collection
        scheduler.start(hour=args.hour, minute=args.minute)
        
        try:
            # Keep running
            while True:
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
