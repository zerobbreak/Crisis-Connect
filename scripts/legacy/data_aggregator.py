#!/usr/bin/env python3
"""
Data Aggregator - Collect and prepare data for model retraining
Pulls weather data from MongoDB and prepares training datasets
"""
import asyncio
import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
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


class DataAggregator:
    """Aggregate and prepare data for model retraining"""
    
    def __init__(self, db_uri: str = None):
        self.db_uri = db_uri or "mongodb://localhost:27017"
        self.db_name = "crisisconnect"
        self.client = None
        self.db = None
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.db_uri)
            self.db = self.client[self.db_name]
            # Test connection
            await self.client.admin.command('ping')
            logger.info("[OK] Connected to MongoDB", database=self.db_name)
            return True
        except Exception as e:
            logger.error("[ERROR] Failed to connect to MongoDB", error=str(e))
            return False
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("[OK] MongoDB connection closed")
    
    async def get_weather_data(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Fetch weather data from MongoDB
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with weather data
        """
        try:
            query = {}
            
            # Add date filter if provided
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date.strftime("%Y-%m-%d %H:%M:%S")
                if end_date:
                    date_filter["$lte"] = end_date.strftime("%Y-%m-%d %H:%M:%S")
                query["timestamp"] = date_filter
            
            logger.info("[DATA] Fetching weather data from MongoDB", query=query)
            
            # Fetch data
            cursor = self.db["weather_data"].find(query, {"_id": 0})
            records = await cursor.to_list(length=None)
            
            if not records:
                logger.warning("[DATA] No weather data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            logger.info(f"[OK] Fetched {len(df)} weather records", 
                       date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error("[ERROR] Failed to fetch weather data", error=str(e), exc_info=True)
            return pd.DataFrame()
    
    async def get_predictions(self, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Fetch predictions from MongoDB
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            DataFrame with predictions
        """
        try:
            query = {}
            
            # Add date filter if provided
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date.strftime("%Y-%m-%d %H:%M:%S")
                if end_date:
                    date_filter["$lte"] = end_date.strftime("%Y-%m-%d %H:%M:%S")
                query["timestamp"] = date_filter
            
            logger.info("[DATA] Fetching predictions from MongoDB", query=query)
            
            # Fetch data
            cursor = self.db["predictions"].find(query, {"_id": 0})
            records = await cursor.to_list(length=None)
            
            if not records:
                logger.warning("[DATA] No predictions found")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            logger.info(f"[OK] Fetched {len(df)} prediction records")
            
            return df
            
        except Exception as e:
            logger.error("[ERROR] Failed to fetch predictions", error=str(e), exc_info=True)
            return pd.DataFrame()
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        initial_count = len(df)
        logger.info(f"[CLEAN] Starting validation with {initial_count} records")
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"[CLEAN] Removed {initial_count - len(df)} duplicates")
        
        # Check for required columns
        required_cols = ['location', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"[CLEAN] Missing required columns: {missing_cols}")
        
        # Remove rows with missing critical values
        before_na = len(df)
        df = df.dropna(subset=required_cols, how='any')
        logger.info(f"[CLEAN] Removed {before_na - len(df)} rows with missing values")
        
        # Validate coordinates
        if 'lat' in df.columns and 'lon' in df.columns:
            before_coords = len(df)
            df = df[(df['lat'].between(-90, 90)) & (df['lon'].between(-180, 180))]
            logger.info(f"[CLEAN] Removed {before_coords - len(df)} rows with invalid coordinates")
        
        logger.info(f"[OK] Validation complete: {len(df)} valid records ({len(df)/initial_count*100:.1f}% retained)")
        
        return df
    
    async def aggregate_training_data(self, 
                                     start_date: datetime = None,
                                     end_date: datetime = None,
                                     output_file: str = None) -> Dict[str, Any]:
        """
        Aggregate all data for training
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            output_file: Path to save aggregated data
            
        Returns:
            Dictionary with aggregation results
        """
        try:
            print("=" * 80)
            print("DATA AGGREGATION FOR MODEL RETRAINING")
            print("=" * 80)
            
            # Fetch weather data
            print("\n[1/3] Fetching weather data...")
            weather_df = await self.get_weather_data(start_date, end_date)
            
            # Fetch predictions
            print("[2/3] Fetching predictions...")
            predictions_df = await self.get_predictions(start_date, end_date)
            
            # Combine datasets
            print("[3/3] Combining and cleaning data...")
            if not weather_df.empty and not predictions_df.empty:
                # Merge on location and timestamp
                combined_df = pd.merge(
                    weather_df, 
                    predictions_df,
                    on=['location', 'timestamp'],
                    how='outer',
                    suffixes=('_weather', '_pred')
                )
            elif not weather_df.empty:
                combined_df = weather_df
            elif not predictions_df.empty:
                combined_df = predictions_df
            else:
                logger.warning("[DATA] No data available for aggregation")
                return {
                    "success": False,
                    "message": "No data available",
                    "record_count": 0
                }
            
            # Clean data
            combined_df = self.validate_and_clean(combined_df)
            
            # Save to file if specified
            if output_file and not combined_df.empty:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                combined_df.to_csv(output_file, index=False)
                logger.info(f"[OK] Saved aggregated data to {output_file}")
                print(f"\n[SAVED] Data exported to: {output_file}")
            
            # Summary
            print("\n" + "=" * 80)
            print("AGGREGATION SUMMARY")
            print("=" * 80)
            print(f"Weather Records:    {len(weather_df)}")
            print(f"Prediction Records: {len(predictions_df)}")
            print(f"Combined Records:   {len(combined_df)}")
            print(f"Date Range:         {start_date or 'All'} to {end_date or 'All'}")
            if output_file:
                print(f"Output File:        {output_file}")
            print("=" * 80)
            
            return {
                "success": True,
                "message": "Data aggregated successfully",
                "record_count": len(combined_df),
                "weather_count": len(weather_df),
                "prediction_count": len(predictions_df),
                "output_file": output_file,
                "date_range": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None
                }
            }
            
        except Exception as e:
            logger.error("[ERROR] Data aggregation failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "message": f"Aggregation failed: {str(e)}",
                "record_count": 0
            }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Data Aggregator for Model Retraining")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days-back", type=int, default=7, 
                       help="Number of days to look back (default: 7)")
    parser.add_argument("--output", type=str, 
                       default="data/training/aggregated_data.csv",
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = datetime.now() - timedelta(days=args.days_back)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()
    
    # Create aggregator
    aggregator = DataAggregator()
    
    try:
        # Connect to database
        if not await aggregator.connect():
            print("[ERROR] Failed to connect to database")
            return
        
        # Aggregate data
        result = await aggregator.aggregate_training_data(
            start_date=start_date,
            end_date=end_date,
            output_file=args.output
        )
        
        if result['success']:
            print(f"\n[SUCCESS] Aggregated {result['record_count']} records")
        else:
            print(f"\n[FAILED] {result['message']}")
            
    finally:
        await aggregator.close()


if __name__ == "__main__":
    asyncio.run(main())
