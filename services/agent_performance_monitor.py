"""
Agent Performance Monitor

Tracks performance of each agent and the ensemble system.
Logs predictions, calculates accuracy metrics, and monitors agent agreement.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger("crisisconnect.agent_performance_monitor")


@dataclass
class PredictionRecord:
    """Record of a single prediction for tracking"""
    timestamp: str
    location: str
    hazard_type: str
    predicted_risk: float
    predicted_level: str
    hours_to_peak: int
    actual_outcome: Optional[bool]  # Did disaster occur?
    actual_severity: Optional[float]  # How severe was it?
    method_breakdown: Dict[str, float]
    primary_method: str
    confidence: float
    was_anomalous: bool
    warnings: List[str]
    processing_time: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AgentPerformanceMonitor:
    """
    Track performance of each agent and the ensemble
    
    Metrics tracked:
    - Accuracy: How often are predictions correct?
    - Speed: How long does each prediction take?
    - Confidence calibration: Are high-confidence predictions more accurate?
    - Agent agreement: Do agents agree with each other?
    - Anomaly detection rate: How often are anomalies flagged?
    """
    
    def __init__(self, log_dir: str = "data"):
        """
        Initialize performance monitor
        
        Args:
            log_dir: Directory to store prediction logs
        """
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "prediction_log.jsonl"
        self.predictions: List[PredictionRecord] = []
        
        # Load existing predictions
        self._load_predictions()
        
        logger.info(f"AgentPerformanceMonitor initialized with {len(self.predictions)} existing records")
    
    def _load_predictions(self):
        """Load existing predictions from log file"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self.predictions.append(PredictionRecord(**data))
            except Exception as e:
                logger.warning(f"Could not load existing predictions: {e}")
    
    def log_prediction(self,
                      location: str,
                      hazard_type: str,
                      prediction_result: Dict,
                      actual_outcome: Optional[bool] = None,
                      actual_severity: Optional[float] = None):
        """
        Log a prediction for later analysis
        
        Args:
            location: Location identifier
            hazard_type: Type of hazard predicted
            prediction_result: Result from AgentOrchestrator.predict()
            actual_outcome: Did a disaster actually occur? (for validation)
            actual_severity: How severe was the actual event? (0-1)
        """
        pred = prediction_result.get("prediction", {})
        
        record = PredictionRecord(
            timestamp=datetime.now().isoformat(),
            location=location,
            hazard_type=hazard_type,
            predicted_risk=pred.get("risk_score", 0.0),
            predicted_level=pred.get("risk_level", "UNKNOWN"),
            hours_to_peak=pred.get("hours_to_peak", 0),
            actual_outcome=actual_outcome,
            actual_severity=actual_severity,
            method_breakdown=pred.get("method_breakdown", {}),
            primary_method=pred.get("primary_method", "unknown"),
            confidence=pred.get("confidence", 0.0),
            was_anomalous=pred.get("is_anomalous", False),
            warnings=pred.get("warnings", []),
            processing_time=prediction_result.get("processing_time_seconds", 0.0)
        )
        
        self.predictions.append(record)
        
        # Append to log file
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        
        logger.debug(f"Logged prediction for {location}: {record.predicted_risk:.1%}")
    
    def update_outcome(self,
                      location: str,
                      timestamp: str,
                      actual_outcome: bool,
                      actual_severity: float = None):
        """
        Update a prediction with actual outcome (for validation)
        
        Args:
            location: Location of the prediction
            timestamp: Timestamp of the prediction to update
            actual_outcome: Did disaster occur?
            actual_severity: How severe was it?
        """
        for pred in self.predictions:
            if pred.location == location and pred.timestamp == timestamp:
                pred.actual_outcome = actual_outcome
                pred.actual_severity = actual_severity
                logger.info(f"Updated outcome for {location} at {timestamp}")
                
                # Rewrite log file
                self._save_all_predictions()
                return
        
        logger.warning(f"Prediction not found for {location} at {timestamp}")
    
    def _save_all_predictions(self):
        """Rewrite all predictions to log file"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'w') as f:
            for pred in self.predictions:
                f.write(json.dumps(pred.to_dict()) + "\n")
    
    def calculate_metrics(self, days: int = None) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            days: Only consider predictions from last N days (None = all)
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predictions
        
        # Filter by date if specified
        if days is not None:
            cutoff = datetime.now().timestamp() - (days * 86400)
            predictions = [
                p for p in predictions 
                if datetime.fromisoformat(p.timestamp).timestamp() > cutoff
            ]
        
        if not predictions:
            return {"error": "No predictions to analyze"}
        
        # Basic counts
        total = len(predictions)
        with_outcomes = [p for p in predictions if p.actual_outcome is not None]
        
        metrics = {
            "period_days": days,
            "total_predictions": total,
            "predictions_with_outcomes": len(with_outcomes),
            "accuracy": self._calculate_accuracy(with_outcomes),
            "speed": self._calculate_speed_metrics(predictions),
            "confidence_calibration": self._calculate_calibration(with_outcomes),
            "agent_agreement": self._calculate_agreement(predictions),
            "method_performance": self._calculate_method_performance(with_outcomes),
            "anomaly_stats": self._calculate_anomaly_stats(predictions),
            "risk_distribution": self._calculate_risk_distribution(predictions),
            "generated_at": datetime.now().isoformat()
        }
        
        return metrics
    
    def _calculate_accuracy(self, predictions: List[PredictionRecord]) -> Dict:
        """Calculate prediction accuracy"""
        if not predictions:
            return {"note": "No validated predictions"}
        
        # Binary accuracy (predicted high risk and disaster occurred, or low risk and no disaster)
        correct = 0
        for p in predictions:
            predicted_high = p.predicted_risk > 0.5
            actual_high = p.actual_outcome
            if predicted_high == actual_high:
                correct += 1
        
        accuracy = correct / len(predictions) if predictions else 0
        
        # Calculate by risk level
        level_accuracy = {}
        for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
            level_preds = [p for p in predictions if p.predicted_level == level]
            if level_preds:
                level_correct = sum(
                    1 for p in level_preds 
                    if (p.predicted_risk > 0.5) == p.actual_outcome
                )
                level_accuracy[level] = level_correct / len(level_preds)
        
        return {
            "overall": accuracy,
            "by_level": level_accuracy,
            "total_validated": len(predictions)
        }
    
    def _calculate_speed_metrics(self, predictions: List[PredictionRecord]) -> Dict:
        """Calculate processing speed metrics"""
        times = [p.processing_time for p in predictions if p.processing_time > 0]
        
        if not times:
            return {"note": "No timing data"}
        
        return {
            "mean_seconds": float(np.mean(times)),
            "median_seconds": float(np.median(times)),
            "p95_seconds": float(np.percentile(times, 95)),
            "max_seconds": float(max(times)),
            "under_5s_pct": sum(1 for t in times if t < 5) / len(times)
        }
    
    def _calculate_calibration(self, predictions: List[PredictionRecord]) -> Dict:
        """
        Calculate confidence calibration
        
        Good calibration: 80% confident predictions are correct 80% of the time
        """
        if not predictions:
            return {"note": "No validated predictions"}
        
        # Group by confidence bins
        bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        calibration = {}
        
        for low, high in bins:
            bin_preds = [p for p in predictions if low <= p.confidence < high]
            if bin_preds:
                correct = sum(
                    1 for p in bin_preds 
                    if (p.predicted_risk > 0.5) == p.actual_outcome
                )
                calibration[f"{low:.0%}-{high:.0%}"] = {
                    "count": len(bin_preds),
                    "accuracy": correct / len(bin_preds),
                    "expected": (low + high) / 2
                }
        
        return calibration
    
    def _calculate_agreement(self, predictions: List[PredictionRecord]) -> Dict:
        """Calculate how often different methods agree"""
        if not predictions:
            return {"note": "No predictions"}
        
        agreements = {}
        
        for p in predictions:
            methods = list(p.method_breakdown.keys())
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    pair = f"{methods[i]}_vs_{methods[j]}"
                    score1 = p.method_breakdown[methods[i]]
                    score2 = p.method_breakdown[methods[j]]
                    
                    # Agreement = 1 - difference
                    agreement = 1 - abs(score1 - score2)
                    
                    if pair not in agreements:
                        agreements[pair] = []
                    agreements[pair].append(agreement)
        
        # Calculate averages
        return {
            pair: {
                "mean_agreement": float(np.mean(scores)),
                "min_agreement": float(min(scores)),
                "count": len(scores)
            }
            for pair, scores in agreements.items()
        }
    
    def _calculate_method_performance(self, predictions: List[PredictionRecord]) -> Dict:
        """Calculate performance of individual methods"""
        if not predictions:
            return {"note": "No validated predictions"}
        
        methods = set()
        for p in predictions:
            methods.update(p.method_breakdown.keys())
        
        performance = {}
        for method in methods:
            method_preds = [
                p for p in predictions 
                if method in p.method_breakdown
            ]
            
            if method_preds:
                # Calculate accuracy for this method's predictions
                correct = sum(
                    1 for p in method_preds
                    if (p.method_breakdown[method] > 0.5) == p.actual_outcome
                )
                
                # Calculate average contribution when primary
                primary_count = sum(1 for p in method_preds if p.primary_method == method)
                
                performance[method] = {
                    "accuracy": correct / len(method_preds),
                    "predictions": len(method_preds),
                    "times_primary": primary_count,
                    "avg_risk_score": float(np.mean([p.method_breakdown[method] for p in method_preds]))
                }
        
        return performance
    
    def _calculate_anomaly_stats(self, predictions: List[PredictionRecord]) -> Dict:
        """Calculate anomaly detection statistics"""
        anomalous = [p for p in predictions if p.was_anomalous]
        
        return {
            "total_anomalies": len(anomalous),
            "anomaly_rate": len(anomalous) / len(predictions) if predictions else 0,
            "anomaly_locations": list(set(p.location for p in anomalous))
        }
    
    def _calculate_risk_distribution(self, predictions: List[PredictionRecord]) -> Dict:
        """Calculate distribution of risk predictions"""
        if not predictions:
            return {"note": "No predictions"}
        
        risks = [p.predicted_risk for p in predictions]
        levels = [p.predicted_level for p in predictions]
        
        return {
            "risk_score": {
                "mean": float(np.mean(risks)),
                "median": float(np.median(risks)),
                "std": float(np.std(risks)),
                "min": float(min(risks)),
                "max": float(max(risks))
            },
            "level_counts": {
                level: levels.count(level)
                for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
            }
        }
    
    def get_recent_predictions(self, n: int = 10) -> List[Dict]:
        """Get most recent predictions"""
        recent = sorted(
            self.predictions, 
            key=lambda x: x.timestamp, 
            reverse=True
        )[:n]
        
        return [p.to_dict() for p in recent]
    
    def get_predictions_for_location(self, location: str) -> List[Dict]:
        """Get all predictions for a specific location"""
        location_preds = [
            p for p in self.predictions 
            if p.location == location
        ]
        
        return [p.to_dict() for p in location_preds]
    
    def generate_report(self, days: int = 30) -> str:
        """Generate a human-readable performance report"""
        metrics = self.calculate_metrics(days)
        
        if "error" in metrics:
            return f"Cannot generate report: {metrics['error']}"
        
        lines = [
            "=" * 60,
            "AGENT PERFORMANCE REPORT",
            f"Period: Last {days} days" if days else "All time",
            f"Generated: {metrics['generated_at']}",
            "=" * 60,
            "",
            f"Total Predictions: {metrics['total_predictions']}",
            f"Validated Predictions: {metrics['predictions_with_outcomes']}",
            "",
            "--- ACCURACY ---"
        ]
        
        acc = metrics.get("accuracy", {})
        if isinstance(acc, dict) and "overall" in acc:
            lines.append(f"Overall Accuracy: {acc['overall']:.1%}")
            if acc.get("by_level"):
                for level, val in acc["by_level"].items():
                    lines.append(f"  {level}: {val:.1%}")
        
        lines.extend([
            "",
            "--- SPEED ---"
        ])
        
        speed = metrics.get("speed", {})
        if isinstance(speed, dict) and "mean_seconds" in speed:
            lines.append(f"Mean Processing Time: {speed['mean_seconds']:.2f}s")
            lines.append(f"95th Percentile: {speed['p95_seconds']:.2f}s")
            lines.append(f"Under 5s: {speed['under_5s_pct']:.1%}")
        
        lines.extend([
            "",
            "--- METHOD PERFORMANCE ---"
        ])
        
        method_perf = metrics.get("method_performance", {})
        if isinstance(method_perf, dict):
            for method, perf in method_perf.items():
                if isinstance(perf, dict):
                    lines.append(
                        f"  {method}: {perf.get('accuracy', 0):.1%} accuracy, "
                        f"{perf.get('times_primary', 0)} times primary"
                    )
        
        lines.extend([
            "",
            "--- ANOMALIES ---"
        ])
        
        anomaly = metrics.get("anomaly_stats", {})
        if isinstance(anomaly, dict):
            lines.append(f"Total Anomalies Detected: {anomaly.get('total_anomalies', 0)}")
            lines.append(f"Anomaly Rate: {anomaly.get('anomaly_rate', 0):.1%}")
        
        lines.extend([
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)
    
    def export_metrics(self, filepath: str, days: int = None):
        """Export metrics to JSON file"""
        metrics = self.calculate_metrics(days)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
