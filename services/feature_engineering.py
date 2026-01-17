"""
Feature Engineering Pipeline

Creates ML-ready features from raw disaster and weather data:
- Temporal features (time-based patterns)
- Geographic features (location-based factors)
- Pattern features (disaster clustering and frequency)
- Aggregation features (rolling windows)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create ML-ready features from raw data"""
    
    # Southern Hemisphere seasons
    SEASON_MAP = {
        12: "summer", 1: "summer", 2: "summer",
        3: "autumn", 4: "autumn", 5: "autumn",
        6: "winter", 7: "winter", 8: "winter",
        9: "spring", 10: "spring", 11: "spring",
    }
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names = []
    
    def engineer_disaster_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for disaster prediction
        
        Args:
            df: Raw disaster records (optionally with weather data)
            
        Returns:
            Feature-engineered dataset ready for ML
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided")
            return df
        
        df = df.copy()
        
        # Ensure date is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        
        df = df.sort_values("date").reset_index(drop=True)
        
        logger.info(f"Engineering features for {len(df)} records")
        initial_cols = len(df.columns)
        
        # Add feature groups
        df = self._add_temporal_features(df)
        df = self._add_geographic_features(df)
        df = self._add_pattern_features(df)
        df = self._add_aggregation_features(df)
        df = self._add_interaction_features(df)
        df = self._add_lag_features(df)
        
        # Encode categorical features
        df = self._encode_categoricals(df)
        
        # Track feature names
        new_cols = len(df.columns) - initial_cols
        self.feature_names = list(df.columns)
        
        logger.info(f"Created {new_cols} new features ({len(df.columns)} total columns)")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        
        if "date" not in df.columns:
            return df
        
        # Basic temporal components
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["date"].dt.quarter
        
        # Season (Southern Hemisphere)
        df["season"] = df["month"].map(self.SEASON_MAP)
        
        # Cyclic encoding for periodicity
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        
        # Time since reference date (for trend)
        reference_date = pd.Timestamp("2010-01-01")
        df["days_since_2010"] = (df["date"] - reference_date).dt.days
        
        # Is end of month (higher flood risk due to rainy season patterns)
        df["is_end_of_month"] = (df["day"] >= 25).astype(int)
        
        # Is rainy season (varies by region, using general Southern Africa pattern)
        df["is_rainy_season"] = df["month"].isin([10, 11, 12, 1, 2, 3]).astype(int)
        
        # Days since last event (per country/region)
        if "country" in df.columns:
            df["days_since_last_event"] = df.groupby("country")["date"].diff().dt.days
            df["days_since_last_event"] = df["days_since_last_event"].fillna(365)
        
        return df
    
    def _add_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return df
        
        # Hemisphere
        df["is_southern_hemisphere"] = (df["latitude"] < 0).astype(int)
        
        # Distance to equator (affects weather patterns)
        df["distance_to_equator"] = abs(df["latitude"])
        
        # Latitude bands
        df["latitude_band"] = pd.cut(
            df["latitude"],
            bins=[-90, -35, -25, -15, 0, 15, 25, 35, 90],
            labels=["extreme_south", "south", "south_tropic", "equatorial_south",
                    "equatorial_north", "north_tropic", "north", "extreme_north"]
        )
        
        # Rough coastal indicator (within ~500km of typical coastline)
        # For Southern Africa, very rough approximation
        df["longitude_normalized"] = (df["longitude"] - df["longitude"].mean()) / df["longitude"].std()
        df["is_likely_coastal"] = (
            ((df["longitude"] > 30) & (df["latitude"] < -20)) |  # East coast
            ((df["longitude"] < 20) & (df["latitude"] < -30))    # West coast
        ).astype(int)
        
        # Elevation proxy (rough estimate based on coordinates)
        # Higher elevation areas in eastern Southern Africa
        df["elevation_proxy"] = (
            (df["latitude"].between(-30, -25)) &
            (df["longitude"].between(28, 32))
        ).astype(int)
        
        # Location clustering (grid-based)
        df["lat_grid"] = (df["latitude"] / 2).round() * 2
        df["lon_grid"] = (df["longitude"] / 2).round() * 2
        df["grid_cell"] = df["lat_grid"].astype(str) + "_" + df["lon_grid"].astype(str)
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create disaster pattern features"""
        
        # Disaster type frequency
        if "disaster_type" in df.columns:
            type_counts = df["disaster_type"].value_counts().to_dict()
            df["disaster_type_frequency"] = df["disaster_type"].map(type_counts)
            df["disaster_type_frequency"] = df["disaster_type_frequency"].fillna(1)
        
        # Average impact by disaster type
        if "deaths" in df.columns and "disaster_type" in df.columns:
            type_deaths = df.groupby("disaster_type")["deaths"].mean().to_dict()
            df["type_avg_deaths"] = df["disaster_type"].map(type_deaths)
            df["type_avg_deaths"] = df["type_avg_deaths"].fillna(0)
        
        if "affected" in df.columns and "disaster_type" in df.columns:
            type_affected = df.groupby("disaster_type")["affected"].mean().to_dict()
            df["type_avg_affected"] = df["disaster_type"].map(type_affected)
            df["type_avg_affected"] = df["type_avg_affected"].fillna(0)
        
        # Location-based event density
        if "country" in df.columns:
            country_counts = df["country"].value_counts().to_dict()
            df["country_event_count"] = df["country"].map(country_counts)
            df["country_event_count"] = df["country_event_count"].fillna(1)
        
        # Events per year by country
        if "country" in df.columns and "year" in df.columns:
            yearly_counts = df.groupby(["country", "year"]).size().reset_index(name="events_per_year")
            df = df.merge(yearly_counts, on=["country", "year"], how="left")
            df["events_per_year"] = df["events_per_year"].fillna(1)
        
        return df
    
    def _add_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window aggregations"""
        
        if "date" not in df.columns:
            return df
        
        df = df.sort_values("date").reset_index(drop=True)
        
        # Simple cumulative count of events (simpler approach for small datasets)
        df["cumulative_events"] = range(1, len(df) + 1)
        
        # For each window size, calculate event count using simple rolling
        for window_days in [7, 14, 30, 90]:
            col_suffix = f"_{window_days}d"
            
            # Simple approach: count events in rolling window by date
            dates = df["date"]
            event_counts = []
            
            for i, current_date in enumerate(dates):
                window_start = current_date - pd.Timedelta(days=window_days)
                count = ((dates >= window_start) & (dates <= current_date)).sum()
                event_counts.append(count)
            
            df[f"event_count{col_suffix}"] = event_counts
            
            # Rolling sum of deaths if available
            if "deaths" in df.columns:
                deaths_sums = []
                for i, current_date in enumerate(dates):
                    window_start = current_date - pd.Timedelta(days=window_days)
                    mask = (dates >= window_start) & (dates <= current_date)
                    deaths_sums.append(df.loc[mask, "deaths"].sum())
                df[f"deaths_sum{col_suffix}"] = deaths_sums
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        
        # Season x Coastal interaction
        if "is_rainy_season" in df.columns and "is_likely_coastal" in df.columns:
            df["rainy_coastal_interaction"] = df["is_rainy_season"] * df["is_likely_coastal"]
        
        # Temperature-precipitation interaction (if weather data available)
        if "temp_mean" in df.columns and "precipitation" in df.columns:
            df["temp_precip_interaction"] = df["temp_mean"] * df["precipitation"]
            df["heat_index"] = df["temp_mean"] + 0.1 * df["precipitation"]
        
        # Severity indicator
        if "deaths" in df.columns and "affected" in df.columns:
            df["total_impact"] = df["deaths"] * 100 + df["affected"]
            df["severity_score"] = np.log1p(df["total_impact"])
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for time series prediction"""
        
        if "country" not in df.columns:
            return df
        
        df = df.sort_values(["country", "date"])
        
        # Lag features by country
        lag_cols = ["deaths", "affected"] if all(c in df.columns for c in ["deaths", "affected"]) else []
        
        for col in lag_cols:
            for lag in [1, 2, 3]:
                df[f"{col}_lag_{lag}"] = df.groupby("country")[col].shift(lag)
                df[f"{col}_lag_{lag}"] = df[f"{col}_lag_{lag}"].fillna(0)
        
        return df
    
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        
        # Season encoding
        if "season" in df.columns:
            season_dummies = pd.get_dummies(df["season"], prefix="season")
            df = pd.concat([df, season_dummies], axis=1)
        
        # Disaster type encoding
        if "disaster_type" in df.columns:
            type_dummies = pd.get_dummies(df["disaster_type"], prefix="disaster")
            df = pd.concat([df, type_dummies], axis=1)
        
        # Latitude band encoding
        if "latitude_band" in df.columns:
            lat_dummies = pd.get_dummies(df["latitude_band"], prefix="lat_band")
            df = pd.concat([df, lat_dummies], axis=1)
        
        return df
    
    def engineer_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from weather data
        
        Args:
            df: Weather data DataFrame
            
        Returns:
            Weather data with engineered features
        """
        if len(df) == 0:
            return df
        
        df = df.copy()
        
        # Ensure date is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # Temperature range
        if all(c in df.columns for c in ["temp_max", "temp_min"]):
            df["temp_range"] = df["temp_max"] - df["temp_min"]
        
        # Temperature anomaly (deviation from mean)
        if "temp_mean" in df.columns:
            monthly_mean = df.groupby(df["date"].dt.month)["temp_mean"].transform("mean")
            df["temp_anomaly"] = df["temp_mean"] - monthly_mean
        
        # Precipitation intensity
        if "precipitation" in df.columns:
            df["is_heavy_rain"] = (df["precipitation"] > 50).astype(int)
            df["is_extreme_rain"] = (df["precipitation"] > 100).astype(int)
        
        # Wind severity
        if "wind_speed_max" in df.columns:
            df["is_high_wind"] = (df["wind_speed_max"] > 60).astype(int)
            df["is_storm_wind"] = (df["wind_speed_max"] > 100).astype(int)
        
        # Rolling weather statistics
        if "date" in df.columns:
            for col in ["precipitation", "temp_mean", "wind_speed_max"]:
                if col in df.columns:
                    # 7-day rolling mean
                    df[f"{col}_7d_mean"] = df[col].rolling(7, min_periods=1).mean()
                    # 7-day rolling max
                    df[f"{col}_7d_max"] = df[col].rolling(7, min_periods=1).max()
                    # 30-day rolling mean
                    df[f"{col}_30d_mean"] = df[col].rolling(30, min_periods=1).mean()
        
        # Cumulative precipitation (for flood prediction)
        if "precipitation" in df.columns:
            df["precip_cumsum_7d"] = df["precipitation"].rolling(7, min_periods=1).sum()
            df["precip_cumsum_14d"] = df["precipitation"].rolling(14, min_periods=1).sum()
            df["precip_cumsum_30d"] = df["precipitation"].rolling(30, min_periods=1).sum()
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names from last engineering run"""
        return self.feature_names
    
    def get_numeric_features(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric feature columns"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
