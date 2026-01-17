"""
Data Quality Monitor

Monitors overall data quality, tracks improvements over time,
and generates reports for the data pipeline.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """Monitor and report on data quality over time"""
    
    def __init__(self, reports_path: str = "data/quality_reports.jsonl"):
        """
        Initialize monitor
        
        Args:
            reports_path: Path to store quality reports
        """
        self.reports_path = Path(reports_path)
        self.reports_path.parent.mkdir(parents=True, exist_ok=True)
    
    def assess_quality(self, df: pd.DataFrame, source: str = "unknown") -> Dict[str, Any]:
        """
        Calculate comprehensive quality score
        
        Args:
            df: DataFrame to assess
            source: Data source identifier
            
        Returns:
            Quality report with scores and recommendations
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "record_count": len(df),
            "column_count": len(df.columns),
            "overall_score": 0,
            "breakdown": {},
            "issues": [],
            "recommendations": [],
        }
        
        if len(df) == 0:
            report["issues"].append("Dataset is empty")
            return report
        
        # Score components
        report["breakdown"]["completeness"] = self._score_completeness(df)
        report["breakdown"]["validity"] = self._score_validity(df)
        report["breakdown"]["consistency"] = self._score_consistency(df)
        report["breakdown"]["accuracy_proxy"] = self._score_accuracy(df)
        
        # Overall score (average)
        scores = list(report["breakdown"].values())
        report["overall_score"] = sum(scores) / len(scores) if scores else 0
        
        # Generate recommendations based on scores
        self._add_recommendations(report)
        
        return report
    
    def _score_completeness(self, df: pd.DataFrame) -> float:
        """Percentage of non-null values"""
        total_cells = len(df) * len(df.columns)
        if total_cells == 0:
            return 0
        
        filled_cells = df.notna().sum().sum()
        return round((filled_cells / total_cells) * 100, 2)
    
    def _score_validity(self, df: pd.DataFrame) -> float:
        """Percentage of values in valid ranges"""
        valid_count = 0
        total_count = 0
        
        # Check numeric ranges
        validations = [
            ("latitude", -90, 90),
            ("longitude", -180, 180),
            ("deaths", 0, 1_000_000),
            ("affected", 0, 1_000_000_000),
            ("temp_max", -60, 60),
            ("temp_min", -80, 60),
            ("precipitation", 0, 1000),
        ]
        
        for col, min_val, max_val in validations:
            if col not in df.columns:
                continue
            
            valid = df[col].between(min_val, max_val) | df[col].isna()
            valid_count += valid.sum()
            total_count += len(df)
        
        if total_count == 0:
            return 100
        
        return round((valid_count / total_count) * 100, 2)
    
    def _score_consistency(self, df: pd.DataFrame) -> float:
        """Check for data consistency"""
        score = 100
        
        # Check date consistency
        if "date" in df.columns:
            if df["date"].dtype != "datetime64[ns]":
                score -= 10
        
        # Check for duplicate primary keys
        id_cols = [c for c in ["disaster_id", "id"] if c in df.columns]
        if id_cols:
            dups = df[id_cols[0]].duplicated().sum()
            if dups > 0:
                dup_rate = dups / len(df) * 100
                score -= min(20, dup_rate)
        
        # Temperature consistency
        if all(c in df.columns for c in ["temp_max", "temp_min"]):
            inconsistent = (df["temp_min"] > df["temp_max"]).sum()
            if inconsistent > 0:
                rate = inconsistent / len(df) * 100
                score -= min(10, rate)
        
        return max(0, round(score, 2))
    
    def _score_accuracy(self, df: pd.DataFrame) -> float:
        """Proxy for accuracy based on obvious errors"""
        score = 100
        
        # Negative impact values
        for col in ["deaths", "affected", "injured"]:
            if col in df.columns:
                neg = (df[col] < 0).sum()
                if neg > 0:
                    score -= 5
        
        # Dates in future
        if "date" in df.columns and df["date"].dtype == "datetime64[ns]":
            future = (df["date"] > datetime.now()).sum()
            if future > 0:
                rate = future / len(df) * 100
                score -= min(10, rate)
        
        return max(0, round(score, 2))
    
    def _add_recommendations(self, report: Dict):
        """Add actionable recommendations"""
        
        if report["overall_score"] < 70:
            report["recommendations"].append(
                "Data quality is below 70%. Review cleaning pipeline."
            )
        
        if report["breakdown"]["completeness"] < 80:
            report["recommendations"].append(
                "Too many missing values. Improve data sources or imputation."
            )
        
        if report["breakdown"]["validity"] < 90:
            report["recommendations"].append(
                "Values outside expected ranges. Check coordinate and numeric parsing."
            )
        
        if report["breakdown"]["consistency"] < 90:
            report["recommendations"].append(
                "Data consistency issues. Check for duplicates and logical errors."
            )
        
        if report["record_count"] < 100:
            report["recommendations"].append(
                "Dataset is small. Consider adding more data sources."
            )
    
    def save_report(self, report: Dict):
        """Append report to history file"""
        with open(self.reports_path, "a") as f:
            f.write(json.dumps(report, default=str) + "\n")
        
        logger.info(
            f"Quality report saved: score={report['overall_score']:.1f}, "
            f"records={report['record_count']}"
        )
    
    def load_history(self) -> List[Dict]:
        """Load all historical reports"""
        if not self.reports_path.exists():
            return []
        
        reports = []
        with open(self.reports_path, "r") as f:
            for line in f:
                try:
                    reports.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return reports
    
    def get_trend(self, n_reports: int = 10) -> Dict[str, Any]:
        """
        Get quality trend from recent reports
        
        Args:
            n_reports: Number of recent reports to analyze
            
        Returns:
            Trend analysis with direction and change
        """
        history = self.load_history()
        
        if len(history) < 2:
            return {"trend": "insufficient_data"}
        
        recent = history[-n_reports:]
        scores = [r.get("overall_score", 0) for r in recent]
        
        # Calculate trend
        if len(scores) >= 2:
            first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
            second_half = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
            change = second_half - first_half
            
            if change > 5:
                direction = "improving"
            elif change < -5:
                direction = "declining"
            else:
                direction = "stable"
        else:
            direction = "unknown"
            change = 0
        
        return {
            "trend": direction,
            "change": round(change, 2),
            "latest_score": scores[-1] if scores else 0,
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "reports_analyzed": len(recent),
        }
    
    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a quick data summary
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Summary statistics
        """
        summary = {
            "total_records": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }
        
        # Date range
        if "date" in df.columns and df["date"].dtype == "datetime64[ns]":
            summary["date_range"] = {
                "start": df["date"].min().isoformat() if pd.notna(df["date"].min()) else None,
                "end": df["date"].max().isoformat() if pd.notna(df["date"].max()) else None,
            }
        
        # Geographic coverage
        if "country" in df.columns:
            summary["countries"] = df["country"].nunique()
        
        # Disaster type distribution
        if "disaster_type" in df.columns:
            summary["disaster_types"] = df["disaster_type"].value_counts().to_dict()
        
        # Impact statistics
        for col in ["deaths", "affected"]:
            if col in df.columns:
                summary[f"{col}_stats"] = {
                    "total": int(df[col].sum()),
                    "mean": round(df[col].mean(), 2),
                    "max": int(df[col].max()),
                }
        
        return summary
