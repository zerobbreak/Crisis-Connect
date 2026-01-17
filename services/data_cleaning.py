"""
Data Cleaning Pipeline

Multi-step cleaning process for disaster and weather data:
1. Remove duplicates (exact and near-duplicates)
2. Normalize column names
3. Parse dates (multiple formats)
4. Validate coordinates
5. Remove outliers
6. Handle missing values intelligently
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and validate disaster data"""
    
    # Standard column name mappings
    COLUMN_MAPPINGS = {
        # EM-DAT style
        "Event Date": "date",
        "Event Name": "event_name",
        "Disaster Type": "disaster_type",
        "Disaster Subtype": "disaster_subtype",
        "Country": "country",
        "Location": "location",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Total Deaths": "deaths",
        "No Deaths": "deaths",
        "Total Affected": "affected",
        "No Affected": "affected",
        "Total Damage": "damage_usd",
        "Total Damage ('000 US$)": "damage_usd",
        
        # Weather data style
        "temp_2m_max": "temp_max",
        "temp_2m_min": "temp_min",
        "temp_2m_mean": "temp_mean",
        "precipitation_sum": "precipitation",
        "rain_sum": "rain",
        "wind_speed_10m_max": "wind_speed_max",
        "wind_gusts_10m_max": "wind_gusts_max",
        
        # Local Excel file style
        "Year": "year",
        "Month": "month",
        "Day": "day",
        "Start Year": "year",
        "Province": "province",
        "Region": "region",
    }
    
    # Valid ranges for numeric fields
    VALID_RANGES = {
        "latitude": (-90, 90),
        "longitude": (-180, 180),
        "deaths": (0, 1_000_000),
        "affected": (0, 1_000_000_000),
        "damage_usd": (0, 1_000_000_000_000),
        "temp_max": (-60, 60),
        "temp_min": (-80, 60),
        "temp_mean": (-70, 55),
        "precipitation": (0, 1000),  # mm per day
        "wind_speed_max": (0, 400),  # km/h
    }
    
    def __init__(self):
        """Initialize cleaner"""
        self.cleaning_report = {}
        self._reset_report()
    
    def _reset_report(self):
        """Reset cleaning report for new run"""
        self.cleaning_report = {
            "input_records": 0,
            "output_records": 0,
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "invalid_coordinates": 0,
            "invalid_dates": 0,
            "outliers_removed": 0,
            "missing_filled": 0,
            "columns_renamed": [],
            "cleaning_steps": [],
        }
    
    def clean_disasters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-step cleaning pipeline for disaster data
        
        Args:
            df: Raw disaster DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self._reset_report()
        df = df.copy()
        
        self.cleaning_report["input_records"] = len(df)
        logger.info(f"Starting cleaning with {len(df)} records")
        
        # Step 1: Duplicates
        df = self._remove_duplicates(df)
        self.cleaning_report["cleaning_steps"].append("duplicates_removed")
        
        # Step 2: Column names
        df = self._normalize_columns(df)
        self.cleaning_report["cleaning_steps"].append("columns_normalized")
        
        # Step 3: Dates
        df = self._parse_dates(df)
        self.cleaning_report["cleaning_steps"].append("dates_parsed")
        
        # Step 4: Coordinates
        df = self._validate_coordinates(df)
        self.cleaning_report["cleaning_steps"].append("coordinates_validated")
        
        # Step 5: Outliers
        df = self._remove_outliers(df)
        self.cleaning_report["cleaning_steps"].append("outliers_removed")
        
        # Step 6: Missing values
        df = self._handle_missing(df)
        self.cleaning_report["cleaning_steps"].append("missing_handled")
        
        # Step 7: Data types
        df = self._fix_data_types(df)
        self.cleaning_report["cleaning_steps"].append("types_fixed")
        
        self.cleaning_report["output_records"] = len(df)
        logger.info(f"Cleaning complete: {len(df)} records remaining")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicates and near-duplicates"""
        
        # Exact duplicates
        before = len(df)
        df = df.drop_duplicates()
        exact_dups = before - len(df)
        self.cleaning_report["exact_duplicates"] = exact_dups
        
        if exact_dups > 0:
            logger.info(f"  Removed {exact_dups} exact duplicates")
        
        # Near duplicates (same date + location + type)
        dedup_cols = []
        for col in ["date", "latitude", "longitude", "disaster_type"]:
            if col in df.columns:
                dedup_cols.append(col)
        
        if len(dedup_cols) >= 2:
            before = len(df)
            df = df.drop_duplicates(subset=dedup_cols, keep="first")
            near_dups = before - len(df)
            self.cleaning_report["near_duplicates"] = near_dups
            
            if near_dups > 0:
                logger.info(f"  Removed {near_dups} near-duplicates")
        
        return df
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        
        # Apply known mappings
        renamed = {}
        for old_name, new_name in self.COLUMN_MAPPINGS.items():
            if old_name in df.columns:
                renamed[old_name] = new_name
        
        if renamed:
            df = df.rename(columns=renamed)
            self.cleaning_report["columns_renamed"] = list(renamed.keys())
        
        # Lowercase and snake_case remaining columns
        df.columns = [
            re.sub(r'[^\w\s]', '', str(col))
            .strip()
            .lower()
            .replace(' ', '_')
            for col in df.columns
        ]
        
        return df
    
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column, handle multiple formats"""
        
        # Check for date column or construct from year/month/day
        if "date" not in df.columns:
            if all(c in df.columns for c in ["year", "month", "day"]):
                df["date"] = pd.to_datetime(
                    df[["year", "month", "day"]],
                    errors="coerce"
                )
            elif "year" in df.columns:
                df["date"] = pd.to_datetime(
                    df["year"].astype(str) + "-01-01",
                    errors="coerce"
                )
            else:
                logger.warning("No date information found")
                return df
        
        # Parse date column
        if df["date"].dtype != "datetime64[ns]":
            date_formats = [
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%Y/%m/%d",
                "%Y%m%d",
                "%d-%m-%Y",
            ]
            
            parsed = None
            for fmt in date_formats:
                try:
                    parsed = pd.to_datetime(df["date"], format=fmt, errors="coerce")
                    if parsed.notna().sum() > len(df) * 0.5:
                        df["date"] = parsed
                        break
                except Exception:
                    continue
            
            # If no format worked, use pandas inference
            if parsed is None or parsed.isna().all():
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # Remove invalid dates
        invalid_dates = df["date"].isna().sum()
        if invalid_dates > 0:
            self.cleaning_report["invalid_dates"] = invalid_dates
            logger.warning(f"  {invalid_dates} records with invalid dates")
        
        # Drop rows with no date (critical field)
        df = df.dropna(subset=["date"])
        
        return df
    
    def _validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with invalid coordinates"""
        
        # Check for latitude/longitude columns (case-insensitive)
        lat_col = None
        lon_col = None
        for col in df.columns:
            if col.lower() == "latitude":
                lat_col = col
            elif col.lower() == "longitude":
                lon_col = col
        
        if not lat_col or not lon_col:
            return df
        
        before = len(df)
        
        # Convert to numeric (ensure we're working with Series)
        # Handle case where column might be duplicated (returns DataFrame)
        try:
            # Get the column as a Series (handle duplicate column names)
            lat_data = df[lat_col]
            if isinstance(lat_data, pd.DataFrame):
                # If duplicate columns, use first one
                lat_series = lat_data.iloc[:, 0]
                # Drop duplicate columns and keep first
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
                lat_col = [c for c in df.columns if c.lower() == 'latitude'][0]
            else:
                lat_series = lat_data
            
            df[lat_col] = pd.to_numeric(lat_series, errors="coerce")
        except Exception as e:
            logger.warning(f"Error converting latitude: {e}, skipping coordinate validation")
            return df
        
        try:
            lon_data = df[lon_col]
            if isinstance(lon_data, pd.DataFrame):
                lon_series = lon_data.iloc[:, 0]
                lon_col = [c for c in df.columns if c.lower() == 'longitude'][0]
            else:
                lon_series = lon_data
            
            df[lon_col] = pd.to_numeric(lon_series, errors="coerce")
        except Exception as e:
            logger.warning(f"Error converting longitude: {e}, skipping coordinate validation")
            return df
        
        # Valid ranges
        valid_mask = (
            df[lat_col].notna() &
            df[lon_col].notna() &
            df[lat_col].between(-90, 90) &
            df[lon_col].between(-180, 180)
        )
        
        invalid = before - valid_mask.sum()
        
        # Keep rows with valid coordinates OR where coords are optional
        # For disaster data, we want coordinates but can keep records without
        df = df[valid_mask | (df[lat_col].isna() & df[lon_col].isna())]
        
        if invalid > 0:
            self.cleaning_report["invalid_coordinates"] = invalid
            logger.info(f"  Flagged {invalid} records with invalid coordinates")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove obvious data entry errors"""
        
        outliers_removed = 0
        
        for col, (min_val, max_val) in self.VALID_RANGES.items():
            if col not in df.columns:
                continue
            
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Count outliers
            outliers = ~df[col].between(min_val, max_val) & df[col].notna()
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                # For critical fields, remove rows; for others, set to NaN
                if col in ["latitude", "longitude"]:
                    df = df[~outliers]
                else:
                    df.loc[outliers, col] = np.nan
                
                outliers_removed += n_outliers
                logger.debug(f"  {col}: {n_outliers} outliers")
        
        self.cleaning_report["outliers_removed"] = outliers_removed
        
        if outliers_removed > 0:
            logger.info(f"  Removed/nullified {outliers_removed} outliers")
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        
        filled = 0
        
        # Impact fields: assume 0 if not reported
        for col in ["deaths", "injured", "affected", "homeless", "damage_usd"]:
            if col in df.columns:
                n_missing = df[col].isna().sum()
                df[col] = df[col].fillna(0)
                filled += n_missing
        
        # Disaster type: flag as "unknown"
        if "disaster_type" in df.columns:
            n_missing = df["disaster_type"].isna().sum()
            df["disaster_type"] = df["disaster_type"].fillna("unknown")
            filled += n_missing
        
        # Country: keep as is (will be handled in feature engineering)
        # Coordinates: keep NaN (location might be known from country)
        
        self.cleaning_report["missing_filled"] = filled
        
        if filled > 0:
            logger.info(f"  Filled {filled} missing values")
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types"""
        
        # Numeric columns
        numeric_cols = [
            "latitude", "longitude",
            "deaths", "injured", "affected", "homeless", "damage_usd",
            "temp_max", "temp_min", "temp_mean",
            "precipitation", "rain", "snowfall",
            "wind_speed_max", "wind_gusts_max",
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Integer columns (for counts)
        int_cols = ["deaths", "injured", "affected", "homeless", "year"]
        for col in int_cols:
            if col in df.columns and df[col].notna().any():
                df[col] = df[col].fillna(0).astype(int)
        
        # String columns
        str_cols = ["country", "location", "disaster_type", "event_name"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df.loc[df[col] == "nan", col] = ""
        
        return df
    
    def clean_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean weather data (simpler pipeline than disasters)
        
        Args:
            df: Raw weather DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Normalize columns
        df = self._normalize_columns(df)
        
        # Parse dates
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        
        # Remove duplicates
        if all(c in df.columns for c in ["date", "latitude", "longitude"]):
            df = df.drop_duplicates(
                subset=["date", "latitude", "longitude"],
                keep="first"
            )
        
        # Remove weather outliers
        df = self._remove_outliers(df)
        
        # Fix data types
        df = self._fix_data_types(df)
        
        return df
    
    def get_cleaning_report(self) -> Dict:
        """Get report from last cleaning run"""
        report = self.cleaning_report.copy()
        
        # Calculate percentages
        if report["input_records"] > 0:
            removed = report["input_records"] - report["output_records"]
            report["removal_rate"] = round(removed / report["input_records"] * 100, 2)
        
        return report
