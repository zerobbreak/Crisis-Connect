"""
Phase 1 Data Pipeline Tests

Tests for all Phase 1 data foundation components:
- Data cleaning pipeline
- Feature engineering
- Data quality monitoring
- Integration tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.data_cleaning import DataCleaner
from services.feature_engineering import FeatureEngineer
from services.data_quality_monitor import DataQualityMonitor
from services.data_validation import DataValidator
from services.data_storage import DataStorage


class TestDataCleaner:
    """Test suite for data cleaning pipeline"""
    
    @pytest.fixture
    def sample_dirty_data(self):
        """Create sample data with known issues"""
        return pd.DataFrame({
            "date": ["2023-01-01", "2023-01-01", "2023-02-01", "invalid", "2023-03-01"],
            "latitude": [10, 10, -95, 20, 25],  # -95 is invalid
            "longitude": [20, 20, 30, 40, 50],
            "disaster_type": ["flood", "flood", "drought", None, "storm"],
            "deaths": [10, 10, -5, 20, 30],  # -5 is invalid
            "country": ["South Africa", "South Africa", "Zimbabwe", "Botswana", "Namibia"],
        })
    
    @pytest.fixture
    def cleaner(self):
        return DataCleaner()
    
    def test_remove_exact_duplicates(self, cleaner, sample_dirty_data):
        """Test that exact duplicates are removed"""
        cleaned = cleaner.clean_disasters(sample_dirty_data)
        report = cleaner.get_cleaning_report()
        
        assert report["exact_duplicates"] == 1
        assert len(cleaned) < len(sample_dirty_data)
    
    def test_invalid_coordinates_removed(self, cleaner):
        """Test that invalid coordinates are handled"""
        data = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "latitude": [-95, 25],  # -95 is invalid
            "longitude": [30, 40],
            "disaster_type": ["flood", "storm"],
        })
        
        cleaned = cleaner.clean_disasters(data)
        
        # Only valid coordinates should remain
        assert len(cleaned) <= 2
        if "latitude" in cleaned.columns:
            assert all(cleaned["latitude"].between(-90, 90))
    
    def test_negative_deaths_handled(self, cleaner):
        """Test that negative death counts are handled"""
        data = pd.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "latitude": [10, 20],
            "longitude": [30, 40],
            "disaster_type": ["flood", "storm"],
            "deaths": [-5, 10],
        })
        
        cleaned = cleaner.clean_disasters(data)
        
        # Negative deaths should be set to NaN and filled with 0
        assert all(cleaned["deaths"] >= 0)
    
    def test_date_parsing(self, cleaner):
        """Test date parsing with multiple formats"""
        data = pd.DataFrame({
            "date": ["2023-01-15", "15/02/2023", "2023/03/20"],
            "latitude": [10, 20, 30],
            "longitude": [40, 50, 60],
            "disaster_type": ["flood", "storm", "drought"],
        })
        
        cleaned = cleaner.clean_disasters(data)
        
        assert cleaned["date"].dtype == "datetime64[ns]"
        assert len(cleaned) >= 1
    
    def test_cleaning_report(self, cleaner, sample_dirty_data):
        """Test that cleaning report is generated"""
        cleaner.clean_disasters(sample_dirty_data)
        report = cleaner.get_cleaning_report()
        
        assert "input_records" in report
        assert "output_records" in report
        assert "cleaning_steps" in report
        assert len(report["cleaning_steps"]) > 0


class TestFeatureEngineer:
    """Test suite for feature engineering"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample clean data for feature engineering"""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        return pd.DataFrame({
            "date": dates,
            "latitude": [-25] * 30,
            "longitude": [28] * 30,
            "country": ["South Africa"] * 30,
            "disaster_type": ["flood"] * 15 + ["drought"] * 15,
            "deaths": list(range(30)),
            "affected": list(range(0, 3000, 100)),
        })
    
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer()
    
    def test_temporal_features_created(self, engineer, sample_data):
        """Test that temporal features are created"""
        featured = engineer.engineer_disaster_features(sample_data)
        
        assert "year" in featured.columns
        assert "month" in featured.columns
        assert "day_of_week" in featured.columns
        assert "season" in featured.columns
        assert "is_rainy_season" in featured.columns
    
    def test_geographic_features_created(self, engineer, sample_data):
        """Test that geographic features are created"""
        featured = engineer.engineer_disaster_features(sample_data)
        
        assert "is_southern_hemisphere" in featured.columns
        assert "distance_to_equator" in featured.columns
    
    def test_pattern_features_created(self, engineer, sample_data):
        """Test that pattern features are created"""
        featured = engineer.engineer_disaster_features(sample_data)
        
        assert "disaster_type_frequency" in featured.columns
        assert "country_event_count" in featured.columns
    
    def test_feature_count_increase(self, engineer, sample_data):
        """Test that features are actually added"""
        initial_cols = len(sample_data.columns)
        featured = engineer.engineer_disaster_features(sample_data)
        
        assert len(featured.columns) > initial_cols
    
    def test_no_excessive_nulls(self, engineer, sample_data):
        """Test that feature engineering doesn't create too many nulls"""
        featured = engineer.engineer_disaster_features(sample_data)
        
        # Calculate null percentage
        total_cells = len(featured) * len(featured.columns)
        null_cells = featured.isnull().sum().sum()
        null_pct = null_cells / total_cells * 100
        
        assert null_pct < 50, f"Too many nulls: {null_pct:.1f}%"


class TestDataValidator:
    """Test suite for data validation"""
    
    @pytest.fixture
    def validator(self):
        return DataValidator()
    
    @pytest.fixture
    def good_data(self):
        """Create high-quality sample data"""
        return pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100),
            "latitude": np.random.uniform(-35, 0, 100),
            "longitude": np.random.uniform(15, 35, 100),
            "disaster_type": np.random.choice(["flood", "drought", "storm"], 100),
            "country": np.random.choice(["South Africa", "Zimbabwe", "Botswana"], 100),
            "deaths": np.random.randint(0, 100, 100),
            "affected": np.random.randint(100, 10000, 100),
        })
    
    @pytest.fixture
    def bad_data(self):
        """Create low-quality sample data"""
        return pd.DataFrame({
            "date": ["not a date"] * 10,
            "latitude": [999] * 10,  # Invalid
            "longitude": [999] * 10,  # Invalid
            "deaths": [-1] * 10,  # Invalid
        })
    
    def test_good_data_scores_high(self, validator, good_data):
        """Test that good data gets a high quality score"""
        report = validator.validate_disasters(good_data)
        
        assert report["quality_score"] > 70
        assert report["is_valid"] == True
        assert len(report["issues"]) == 0
    
    def test_bad_data_scores_low(self, validator, bad_data):
        """Test that bad data gets a low quality score"""
        report = validator.validate_disasters(bad_data)
        
        # Should have issues
        assert len(report["issues"]) > 0 or len(report["warnings"]) > 0
    
    def test_empty_data_handled(self, validator):
        """Test that empty data is handled"""
        report = validator.validate_disasters(pd.DataFrame())
        
        assert report["is_valid"] == False
        assert "empty" in report["issues"][0].lower()


class TestDataQualityMonitor:
    """Test suite for quality monitoring"""
    
    @pytest.fixture
    def monitor(self, tmp_path):
        return DataQualityMonitor(str(tmp_path / "quality_reports.jsonl"))
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=50),
            "latitude": np.random.uniform(-30, -20, 50),
            "longitude": np.random.uniform(25, 35, 50),
            "deaths": np.random.randint(0, 50, 50),
        })
    
    def test_quality_assessment(self, monitor, sample_data):
        """Test quality assessment generates report"""
        report = monitor.assess_quality(sample_data)
        
        assert "overall_score" in report
        assert "breakdown" in report
        assert report["overall_score"] >= 0
        assert report["overall_score"] <= 100
    
    def test_report_saving(self, monitor, sample_data):
        """Test that reports are saved correctly"""
        report = monitor.assess_quality(sample_data)
        monitor.save_report(report)
        
        history = monitor.load_history()
        assert len(history) == 1
        assert history[0]["overall_score"] == report["overall_score"]
    
    def test_trend_analysis(self, monitor, sample_data):
        """Test trend analysis with multiple reports"""
        # Save multiple reports
        for _ in range(5):
            report = monitor.assess_quality(sample_data)
            monitor.save_report(report)
        
        trend = monitor.get_trend()
        assert "trend" in trend
        assert "latest_score" in trend


class TestDataStorage:
    """Test suite for data storage"""
    
    @pytest.fixture
    def storage(self, tmp_path):
        return DataStorage(str(tmp_path))
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "date": ["2023-01-01"],
            "value": [100],
        })
    
    def test_save_and_load_raw(self, storage, sample_df):
        """Test saving and loading raw data"""
        storage.save_raw(sample_df, "test_source")
        loaded = storage.load_raw("test_source")
        
        assert len(loaded) == len(sample_df)
    
    def test_save_processed(self, storage, sample_df):
        """Test saving processed data"""
        path = storage.save_processed(sample_df, "test_dataset")
        
        assert path.exists()
        
        loaded = storage.load_processed("test_dataset")
        assert len(loaded) == len(sample_df)


class TestEndToEndPipeline:
    """Integration tests for the full pipeline"""
    
    def test_full_pipeline(self, tmp_path):
        """Test the complete data pipeline from raw to featured"""
        # Create raw data
        raw_data = pd.DataFrame({
            "Event Date": ["2023-01-01", "2023-01-01", "2023-02-15"],
            "Latitude": [10, -95, 15],  # One invalid
            "Longitude": [20, 30, 35],
            "Disaster Type": ["Flood", "Flood", "Drought"],
            "Total Deaths": [10, -5, 3],  # One invalid
            "Country": ["South Africa", "Zimbabwe", "Botswana"],
        })
        
        # Step 1: Clean
        cleaner = DataCleaner()
        cleaned = cleaner.clean_disasters(raw_data)
        
        assert len(cleaned) >= 1
        
        # Step 2: Engineer features
        engineer = FeatureEngineer()
        featured = engineer.engineer_disaster_features(cleaned)
        
        assert len(featured.columns) > len(raw_data.columns)
        
        # Step 3: Validate quality
        validator = DataValidator()
        report = validator.validate_disasters(featured)
        
        assert "quality_score" in report
        assert report["quality_score"] >= 0
    
    def test_weather_pipeline(self):
        """Test weather data processing pipeline"""
        # Create sample weather data
        weather_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=30),
            "latitude": [-25.0] * 30,
            "longitude": [28.0] * 30,
            "temp_max": np.random.uniform(25, 35, 30),
            "temp_min": np.random.uniform(15, 25, 30),
            "precipitation": np.random.uniform(0, 50, 30),
        })
        
        # Clean
        cleaner = DataCleaner()
        cleaned = cleaner.clean_weather(weather_data)
        
        assert len(cleaned) == 30
        
        # Feature engineering
        engineer = FeatureEngineer()
        featured = engineer.engineer_weather_features(cleaned)
        
        assert "temp_range" in featured.columns
        assert "precip_cumsum_7d" in featured.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
