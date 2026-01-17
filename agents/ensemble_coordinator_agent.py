"""
Ensemble Coordinator Agent

Coordinates all other agents and combines their predictions into a final decision.
Detects disagreements, generates recommendations, and provides unified output.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from agents.forecast_agent import Forecast
from agents.anomaly_detection_agent import AnomalyResult

logger = logging.getLogger("crisisconnect.agents.ensemble_coordinator")


@dataclass
class EnsembleDecision:
    """Final decision from all agents"""
    risk_score: float  # 0-1, weighted from all agents
    risk_level: str  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    hours_to_peak: int
    primary_method: str  # Which agent was most confident?
    method_breakdown: Dict[str, float]  # Contribution of each method
    confidence: float  # Overall confidence in the decision
    warnings: List[str]  # Disagreement or anomaly warnings
    recommendation: str  # Action to take
    is_anomalous: bool  # Did anomaly detector flag this?
    anomaly_details: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "hours_to_peak": self.hours_to_peak,
            "primary_method": self.primary_method,
            "method_breakdown": self.method_breakdown,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
            "is_anomalous": self.is_anomalous,
            "anomaly_details": self.anomaly_details,
            "timestamp": self.timestamp.isoformat()
        }


class EnsembleCoordinatorAgent:
    """
    Agent that coordinates all other agents and combines predictions
    
    Instead of trusting one agent, combine their predictions:
    - LSTM: "Risk = 65%"
    - Pattern: "Risk = 80%"
    - Progression: "Risk = 70%"
    - Decision: "Risk = 72%" (weighted average)
    
    Also flags disagreements:
    - If LSTM says 20% but Progression says 90%, something's unusual
    - Alerts human to investigate
    
    Handles anomalies:
    - If anomaly detector flags unusual conditions, increases uncertainty
    - May recommend human review
    """
    
    def __init__(self, 
                 weights: Dict[str, float] = None,
                 disagreement_threshold: float = 0.3):
        """
        Args:
            weights: How much to trust each method
                Default: {"lstm": 0.4, "pattern_matching": 0.30, "progression": 0.30}
            disagreement_threshold: Max acceptable difference between methods (0-1)
        """
        self.weights = weights or {
            "lstm": 0.40,
            "pattern_matching": 0.30,
            "progression": 0.30,
            "heuristic": 0.10  # Lower weight for fallback
        }
        self.disagreement_threshold = disagreement_threshold
        
        # Risk level thresholds
        self.risk_thresholds = {
            "LOW": 0.30,
            "MODERATE": 0.55,
            "HIGH": 0.75
            # >= 0.75 is CRITICAL
        }
        
        logger.info(f"EnsembleCoordinatorAgent initialized: weights={self.weights}")
    
    def coordinate(self, 
                  forecasts: List[Forecast],
                  anomaly_result: Optional[AnomalyResult] = None,
                  hazard_type: str = "flood") -> EnsembleDecision:
        """
        Combine multiple forecasts into final decision
        
        Args:
            forecasts: List of Forecast objects from ForecastAgent
            anomaly_result: Optional AnomalyResult from AnomalyDetectionAgent
            hazard_type: Type of hazard being assessed
            
        Returns:
            EnsembleDecision with combined assessment
        """
        if not forecasts:
            return EnsembleDecision(
                risk_score=0.0,
                risk_level="LOW",
                hours_to_peak=999,
                primary_method="none",
                method_breakdown={},
                confidence=0.0,
                warnings=["No forecasts available"],
                recommendation="Unable to assess risk - no data available",
                is_anomalous=False
            )
        
        # Extract components from forecasts
        method_scores = {}
        method_times = {}
        method_confidences = {}
        
        for forecast in forecasts:
            method = forecast.method
            method_scores[method] = forecast.risk_score
            method_times[method] = forecast.hours_to_peak
            method_confidences[method] = forecast.confidence
        
        # Calculate weighted risk score
        weighted_risk = self._calculate_weighted_risk(method_scores, method_confidences)
        
        # Classify risk level
        risk_level = self._classify_risk_level(weighted_risk)
        
        # Estimate peak time (confidence-weighted average)
        peak_time = self._estimate_peak_time(method_times, method_confidences)
        
        # Detect disagreement between methods
        warnings = self._detect_disagreement(method_scores, method_confidences)
        
        # Handle anomaly detection
        is_anomalous = False
        anomaly_details = None
        if anomaly_result:
            is_anomalous = anomaly_result.is_anomalous
            if is_anomalous:
                warnings.append(f"ANOMALY: {anomaly_result.explanation}")
                anomaly_details = {
                    "score": anomaly_result.anomaly_score,
                    "percentile": anomaly_result.percentile,
                    "contributing_features": anomaly_result.contributing_features
                }
                # Increase uncertainty when anomalous
                weighted_risk = min(1.0, weighted_risk + 0.1)
        
        # Find primary method (highest confidence)
        if method_confidences:
            primary_method = max(method_confidences.items(), key=lambda x: x[1])[0]
        else:
            primary_method = forecasts[0].method if forecasts else "none"
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(method_confidences, warnings, is_anomalous)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            weighted_risk, 
            risk_level, 
            peak_time,
            warnings,
            is_anomalous,
            hazard_type
        )
        
        return EnsembleDecision(
            risk_score=weighted_risk,
            risk_level=risk_level,
            hours_to_peak=peak_time,
            primary_method=primary_method,
            method_breakdown=method_scores,
            confidence=confidence,
            warnings=warnings,
            recommendation=recommendation,
            is_anomalous=is_anomalous,
            anomaly_details=anomaly_details
        )
    
    def _calculate_weighted_risk(self, 
                                scores: Dict[str, float],
                                confidences: Dict[str, float]) -> float:
        """
        Calculate weighted average risk score
        
        Weights are adjusted by confidence - more confident methods get more weight
        """
        if not scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, score in scores.items():
            base_weight = self.weights.get(method, 0.1)
            confidence = confidences.get(method, 0.5)
            
            # Adjust weight by confidence
            adjusted_weight = base_weight * confidence
            
            weighted_sum += score * adjusted_weight
            total_weight += adjusted_weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Map risk score to categorical level"""
        if risk_score < self.risk_thresholds["LOW"]:
            return "LOW"
        elif risk_score < self.risk_thresholds["MODERATE"]:
            return "MODERATE"
        elif risk_score < self.risk_thresholds["HIGH"]:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _estimate_peak_time(self, 
                           times: Dict[str, int],
                           confidences: Dict[str, float]) -> int:
        """Estimate peak time, weighted by confidence"""
        if not times:
            return 48  # Default 2 days
        
        total_confidence = sum(confidences.get(m, 0.5) for m in times.keys())
        if total_confidence == 0:
            return int(np.mean(list(times.values())))
        
        weighted_time = sum(
            times[method] * confidences.get(method, 0.5) / total_confidence
            for method in times.keys()
        )
        
        return max(1, int(weighted_time))
    
    def _detect_disagreement(self, 
                            scores: Dict[str, float],
                            confidences: Dict[str, float]) -> List[str]:
        """Find cases where agents strongly disagree"""
        warnings = []
        
        if len(scores) < 2:
            return warnings
        
        score_values = list(scores.values())
        score_range = max(score_values) - min(score_values)
        
        # If range exceeds threshold and methods have reasonable confidence
        if score_range > self.disagreement_threshold:
            # Check if disagreeing methods have high confidence
            high_conf_methods = [
                method for method, conf in confidences.items() 
                if conf > 0.6
            ]
            
            if len(high_conf_methods) > 1:
                # Find which methods disagree
                method_list = list(scores.keys())
                max_method = max(scores.items(), key=lambda x: x[1])[0]
                min_method = min(scores.items(), key=lambda x: x[1])[0]
                
                warnings.append(
                    f"Method disagreement: {max_method} predicts {scores[max_method]:.1%} "
                    f"but {min_method} predicts {scores[min_method]:.1%}"
                )
        
        return warnings
    
    def _calculate_overall_confidence(self,
                                     confidences: Dict[str, float],
                                     warnings: List[str],
                                     is_anomalous: bool) -> float:
        """Calculate overall confidence in the ensemble decision"""
        if not confidences:
            return 0.3
        
        # Start with average confidence
        base_confidence = np.mean(list(confidences.values()))
        
        # Reduce confidence for disagreements
        disagreement_penalty = len([w for w in warnings if "disagreement" in w.lower()]) * 0.1
        
        # Reduce confidence for anomalies
        anomaly_penalty = 0.15 if is_anomalous else 0
        
        # Calculate final confidence
        final_confidence = base_confidence - disagreement_penalty - anomaly_penalty
        
        return max(0.1, min(1.0, final_confidence))
    
    def _generate_recommendation(self,
                                risk_score: float,
                                risk_level: str,
                                hours_to_peak: int,
                                warnings: List[str],
                                is_anomalous: bool,
                                hazard_type: str) -> str:
        """Generate actionable recommendation based on assessment"""
        
        # Base recommendations by risk level
        if risk_level == "LOW":
            recommendation = "Continue routine monitoring. No immediate action required."
        
        elif risk_level == "MODERATE":
            recommendation = (
                f"Prepare emergency systems. "
                f"Peak {hazard_type} conditions expected in ~{hours_to_peak} hours. "
                f"Monitor weather updates and review emergency plans."
            )
        
        elif risk_level == "HIGH":
            recommendation = (
                f"ALERT: Activate emergency protocols. "
                f"Peak {hazard_type} conditions expected in {hours_to_peak} hours. "
                f"Prepare evacuation routes, alert emergency services, "
                f"and notify at-risk communities."
            )
        
        else:  # CRITICAL
            recommendation = (
                f"EMERGENCY: Immediate action required. "
                f"Peak {hazard_type} conditions in {hours_to_peak} hours. "
                f"Begin evacuation procedures, mobilize all emergency resources, "
                f"and activate emergency shelters."
            )
        
        # Add warnings context
        if warnings:
            if is_anomalous:
                recommendation += " NOTE: Unusual conditions detected - consider expert review."
            elif any("disagreement" in w.lower() for w in warnings):
                recommendation += " NOTE: Prediction methods show some disagreement - monitor closely."
        
        return recommendation
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update method weights based on performance feedback"""
        for method, weight in new_weights.items():
            if method in self.weights:
                self.weights[method] = weight
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def get_method_contribution(self, decision: EnsembleDecision) -> Dict[str, float]:
        """
        Calculate how much each method contributed to the final decision
        
        Returns:
            Dict mapping method name to contribution percentage
        """
        if not decision.method_breakdown:
            return {}
        
        contributions = {}
        total_weighted = 0.0
        
        for method, score in decision.method_breakdown.items():
            weight = self.weights.get(method, 0.1)
            contribution = score * weight
            contributions[method] = contribution
            total_weighted += contribution
        
        # Normalize to percentages
        if total_weighted > 0:
            contributions = {k: (v/total_weighted)*100 for k, v in contributions.items()}
        
        return contributions
    
    def explain_decision(self, decision: EnsembleDecision, forecasts: List[Forecast]) -> str:
        """
        Generate detailed explanation of how the decision was made
        
        Args:
            decision: The EnsembleDecision
            forecasts: Original forecasts that went into the decision
            
        Returns:
            Human-readable explanation
        """
        lines = [
            f"=== Risk Assessment Summary ===",
            f"Overall Risk: {decision.risk_score:.1%} ({decision.risk_level})",
            f"Time to Peak: {decision.hours_to_peak} hours",
            f"Confidence: {decision.confidence:.1%}",
            "",
            "=== Method Breakdown ==="
        ]
        
        for forecast in forecasts:
            lines.append(
                f"  {forecast.method}: {forecast.risk_score:.1%} risk "
                f"(confidence: {forecast.confidence:.1%})"
            )
            lines.append(f"    Reasoning: {forecast.reasoning}")
        
        if decision.warnings:
            lines.append("")
            lines.append("=== Warnings ===")
            for warning in decision.warnings:
                lines.append(f"  - {warning}")
        
        if decision.is_anomalous and decision.anomaly_details:
            lines.append("")
            lines.append("=== Anomaly Details ===")
            lines.append(f"  Percentile: {decision.anomaly_details.get('percentile', 'N/A'):.0f}%")
            if decision.anomaly_details.get('contributing_features'):
                lines.append("  Unusual factors:")
                for feature, details in decision.anomaly_details['contributing_features'].items():
                    lines.append(f"    - {feature}: {details}")
        
        lines.append("")
        lines.append("=== Recommendation ===")
        lines.append(decision.recommendation)
        
        return "\n".join(lines)
