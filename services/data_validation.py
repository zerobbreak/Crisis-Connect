"""
Data Validation System

Validates data quality and generates comprehensive quality reports.
Checks: completeness, validity, consistency, and accuracy proxies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and generate reports"""
    
    # Required fields for disaster data
    REQUIRED_FIELDS = ["date", "disaster_type"]
    
    # Recommended fields
    RECOMMENDED_FIELDS = ["latitude", "longitude", "country", "deaths", "affected"]
    
    # Field type specifications
    FIELD_TYPES = {
        "date": "datetime64[ns]",
        "latitude": "float64",
        "longitude": "float64",
        "deaths": "int64",
        "affected": "int64",
        "damage_usd": "float64",
    }
    
    def __init__(self):
        """Initialize validator"""
        pass
    
    def validate_disasters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive validation report for disaster data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with:
            - is_valid: bool
            - quality_score: 0-100
            - issues: List of critical issues
            - warnings: List of non-critical issues
            - breakdown: Score by category
            - recommendations: Actions to improve quality
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "issues": [],
            "warnings": [],
            "quality_score": 100,
            "breakdown": {},
            "field_coverage": {},
            "recommendations": [],
        }
        
        if len(df) == 0:
            report["issues"].append("Dataset is empty")
            report["quality_score"] = 0
            report["is_valid"] = False
            return report
        
        # Score components
        report["breakdown"]["completeness"] = self._score_completeness(df, report)
        report["breakdown"]["validity"] = self._score_validity(df, report)
        report["breakdown"]["consistency"] = self._score_consistency(df, report)
        report["breakdown"]["accuracy_proxy"] = self._score_accuracy(df, report)
        report["breakdown"]["coverage"] = self._score_coverage(df, report)
        
        # Calculate overall score (weighted average)
        weights = {
            "completeness": 0.25,
            "validity": 0.25,
            "consistency": 0.20,
            "accuracy_proxy": 0.15,
            "coverage": 0.15,
        }
        
        report["quality_score"] = sum(
            report["breakdown"][k] * weights[k]
            for k in weights
        )
        
        # Determine validity
        report["is_valid"] = (
            report["quality_score"] >= 60 and
            len(report["issues"]) == 0
        )
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _score_completeness(self, df: pd.DataFrame, report: Dict) -> float:
        """Score based on field presence and non-null values"""
        
        total_possible = len(df) * len(df.columns)
        filled = df.notna().sum().sum()
        
        # Base completeness
        completeness = (filled / total_possible) * 100 if total_possible > 0 else 0
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in df.columns:
                report["issues"].append(f"Missing required field: {field}")
                completeness -= 10
            elif df[field].isna().sum() > len(df) * 0.1:
                pct = df[field].isna().sum() / len(df) * 100
                report["warnings"].append(f"{field}: {pct:.1f}% missing values")
                completeness -= 5
        
        # Field coverage tracking
        for col in df.columns:
            non_null = df[col].notna().sum()
            report["field_coverage"][col] = {
                "present": non_null,
                "missing": len(df) - non_null,
                "coverage_pct": round(non_null / len(df) * 100, 2),
            }
        
        return max(0, min(100, completeness))
    
    def _score_validity(self, df: pd.DataFrame, report: Dict) -> float:
        """Score based on values being in valid ranges"""
        
        valid_count = 0
        total_checks = 0
        
        # Coordinate ranges
        if "latitude" in df.columns:
            valid = df["latitude"].between(-90, 90) | df["latitude"].isna()
            invalid = (~valid).sum()
            if invalid > 0:
                report["warnings"].append(f"latitude: {invalid} values out of range")
            valid_count += valid.sum()
            total_checks += len(df)
        
        if "longitude" in df.columns:
            valid = df["longitude"].between(-180, 180) | df["longitude"].isna()
            invalid = (~valid).sum()
            if invalid > 0:
                report["warnings"].append(f"longitude: {invalid} values out of range")
            valid_count += valid.sum()
            total_checks += len(df)
        
        # Non-negative counts
        for col in ["deaths", "affected", "injured", "homeless"]:
            if col in df.columns:
                valid = (df[col] >= 0) | df[col].isna()
                invalid = (~valid).sum()
                if invalid > 0:
                    report["warnings"].append(f"{col}: {invalid} negative values")
                valid_count += valid.sum()
                total_checks += len(df)
        
        # Date validity
        if "date" in df.columns:
            if df["date"].dtype == "datetime64[ns]":
                # Check reasonable date range (1900-2030)
                valid = df["date"].dt.year.between(1900, 2030) | df["date"].isna()
                invalid = (~valid).sum()
                if invalid > 0:
                    report["warnings"].append(f"date: {invalid} values outside 1900-2030")
                valid_count += valid.sum()
                total_checks += len(df)
            else:
                report["issues"].append("date column not parsed as datetime")
                total_checks += len(df)
        
        return (valid_count / total_checks * 100) if total_checks > 0 else 100
    
    def _score_consistency(self, df: pd.DataFrame, report: Dict) -> float:
        """Score based on data consistency"""
        
        score = 100
        
        # Check year consistency with date
        if "date" in df.columns and "year" in df.columns:
            if df["date"].dtype == "datetime64[ns]":
                mismatch = (df["date"].dt.year != df["year"]).sum()
                if mismatch > 0:
                    pct = mismatch / len(df) * 100
                    report["warnings"].append(f"year/date mismatch: {mismatch} records ({pct:.1f}%)")
                    score -= min(20, pct / 2)
        
        # Check coordinate-country consistency (basic check)
        # Could be enhanced with actual country boundaries
        if all(c in df.columns for c in ["latitude", "country"]):
            # Southern hemisphere check for Southern Africa
            south_africa = df["country"].str.lower().str.contains("south africa", na=False)
            if south_africa.any():
                wrong_hemisphere = south_africa & (df["latitude"] > 0)
                if wrong_hemisphere.any():
                    report["warnings"].append(
                        f"coordinate inconsistency: {wrong_hemisphere.sum()} South Africa records in northern hemisphere"
                    )
                    score -= 5
        
        # Check for duplicate disaster IDs
        if "disaster_id" in df.columns:
            dups = df["disaster_id"].duplicated().sum()
            if dups > 0:
                report["warnings"].append(f"duplicate disaster IDs: {dups}")
                score -= 5
        
        return max(0, score)
    
    def _score_accuracy(self, df: pd.DataFrame, report: Dict) -> float:
        """Score based on proxy accuracy checks"""
        
        score = 100
        
        # Check for suspiciously round numbers (might indicate estimates)
        for col in ["deaths", "affected"]:
            if col not in df.columns:
                continue
            
            non_zero = df[df[col] > 0][col]
            if len(non_zero) > 10:
                # Check how many are round numbers (divisible by 100)
                round_nums = (non_zero % 100 == 0).sum()
                round_pct = round_nums / len(non_zero) * 100
                
                if round_pct > 80:
                    report["warnings"].append(
                        f"{col}: {round_pct:.0f}% are round numbers (may be estimates)"
                    )
                    # Don't penalize much - estimates are normal for disasters
        
        # Check for impossible values
        if "deaths" in df.columns and "affected" in df.columns:
            # Deaths shouldn't exceed affected
            impossible = df["deaths"] > df["affected"]
            impossible = impossible & (df["affected"] > 0)
            if impossible.sum() > 0:
                report["warnings"].append(
                    f"deaths > affected: {impossible.sum()} records"
                )
                score -= 5
        
        return max(0, score)
    
    def _score_coverage(self, df: pd.DataFrame, report: Dict) -> float:
        """Score based on geographic and temporal coverage"""
        
        score = 100
        
        # Geographic coverage
        if "country" in df.columns:
            countries = df["country"].nunique()
            if countries < 3:
                report["warnings"].append(f"Low geographic diversity: {countries} countries")
                score -= 10
            elif countries < 5:
                score -= 5
        
        # Temporal coverage
        if "date" in df.columns and df["date"].dtype == "datetime64[ns]":
            date_range = df["date"].max() - df["date"].min()
            years = date_range.days / 365
            
            if years < 1:
                report["warnings"].append(f"Limited time span: {years:.1f} years")
                score -= 20
            elif years < 5:
                report["warnings"].append(f"Short time span: {years:.1f} years")
                score -= 10
        
        # Disaster type coverage
        if "disaster_type" in df.columns:
            types = df["disaster_type"].nunique()
            if types < 2:
                report["warnings"].append(f"Low disaster type diversity: {types} types")
                score -= 10
        
        return max(0, score)
    
    def _generate_recommendations(self, report: Dict):
        """Generate actionable recommendations"""
        
        if report["quality_score"] < 50:
            report["recommendations"].append(
                "Data quality is critically low. Consider reviewing data sources and collection process."
            )
        
        if report["breakdown"]["completeness"] < 70:
            report["recommendations"].append(
                "Improve data completeness by filling missing values or using more complete data sources."
            )
        
        if report["breakdown"]["validity"] < 80:
            report["recommendations"].append(
                "Review and fix out-of-range values. Check coordinate and date parsing."
            )
        
        if report["breakdown"]["coverage"] < 70:
            report["recommendations"].append(
                "Expand data collection to more countries and longer time periods."
            )
        
        if len(report["issues"]) > 0:
            report["recommendations"].insert(0, 
                "Address critical issues before using this data for analysis."
            )
    
    def validate_weather(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate weather data
        
        Args:
            df: Weather DataFrame
            
        Returns:
            Validation report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "issues": [],
            "warnings": [],
            "quality_score": 100,
        }
        
        if len(df) == 0:
            report["issues"].append("Dataset is empty")
            report["quality_score"] = 0
            report["is_valid"] = False
            return report
        
        # Check for required fields
        required = ["date", "latitude", "longitude"]
        for field in required:
            if field not in df.columns:
                report["issues"].append(f"Missing required field: {field}")
                report["quality_score"] -= 20
        
        # Check for weather measurements
        weather_fields = ["temp_max", "temp_min", "precipitation", "wind_speed_max"]
        present = [f for f in weather_fields if f in df.columns]
        if len(present) < 2:
            report["warnings"].append(f"Limited weather data: only {present}")
            report["quality_score"] -= 10
        
        # Check for temperature consistency
        if "temp_max" in df.columns and "temp_min" in df.columns:
            inconsistent = (df["temp_min"] > df["temp_max"]).sum()
            if inconsistent > 0:
                report["warnings"].append(f"temp_min > temp_max: {inconsistent} records")
                report["quality_score"] -= 5
        
        report["is_valid"] = report["quality_score"] >= 60
        return report
    
    def save_report(self, report: Dict, filepath: str = "data/quality_reports.jsonl"):
        """
        Append report to quality history
        
        Args:
            report: Validation report
            filepath: Path to report file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "a") as f:
            f.write(json.dumps(report, default=str) + "\n")
        
        logger.info(f"Quality score: {report['quality_score']:.1f}")
    
    def load_report_history(self, filepath: str = "data/quality_reports.jsonl") -> List[Dict]:
        """Load all historical reports"""
        
        if not Path(filepath).exists():
            return []
        
        reports = []
        with open(filepath, "r") as f:
            for line in f:
                try:
                    reports.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return reports
