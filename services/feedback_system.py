# services/feedback_system.py
"""
Feedback System Service - Phase 3

Collects outcomes and uses them to improve prediction models.
Implements a virtuous cycle:
1. Make prediction
2. Actions taken
3. Event occurs (or doesn't)
4. Collect actual outcome
5. Feed back into model
6. Improve prediction accuracy
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger("crisisconnect.feedback_system")


@dataclass
class PredictionOutcome:
    """Track what actually happened after a prediction"""
    # Prediction identification
    prediction_id: str
    alert_id: str
    location: str
    hazard_type: str
    prediction_timestamp: str
    
    # What was predicted
    predicted_risk_level: str
    predicted_risk_score: float
    predicted_peak_hours: int
    predicted_severity: str
    method_breakdown: Dict[str, float]
    
    # What actually happened
    actual_disaster_occurred: bool
    actual_peak_time: Optional[str] = None  # ISO timestamp
    actual_severity: Optional[str] = None
    actual_damage_estimate: float = 0.0     # In currency
    actual_affected_population: int = 0
    actual_casualties: int = 0
    actual_injuries: int = 0
    
    # Action effectiveness
    actions_planned: int = 0
    actions_executed: int = 0
    actions_successful: int = 0
    evacuation_ordered: bool = False
    evacuation_rate: float = 0.0           # Percent of at-risk people evacuated
    shelter_utilization: float = 0.0       # Percent of shelter capacity used
    
    # Timing metrics
    lead_time_hours: int = 0               # How much warning time was given
    lead_time_used_hours: int = 0          # Actual hours before peak they acted
    
    # Assessment
    false_alarm: bool = False              # Predicted disaster that didn't happen
    missed_event: bool = False             # Disaster that wasn't predicted
    confidence_rating: float = 0.0         # Did emergency managers trust it?
    
    # Notes
    notes: str = ""
    recorded_by: str = ""
    recorded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "prediction_id": self.prediction_id,
            "alert_id": self.alert_id,
            "location": self.location,
            "hazard_type": self.hazard_type,
            "prediction_timestamp": self.prediction_timestamp,
            "predicted": {
                "risk_level": self.predicted_risk_level,
                "risk_score": self.predicted_risk_score,
                "peak_hours": self.predicted_peak_hours,
                "severity": self.predicted_severity,
                "method_breakdown": self.method_breakdown
            },
            "actual": {
                "disaster_occurred": self.actual_disaster_occurred,
                "peak_time": self.actual_peak_time,
                "severity": self.actual_severity,
                "damage_estimate": self.actual_damage_estimate,
                "affected_population": self.actual_affected_population,
                "casualties": self.actual_casualties,
                "injuries": self.actual_injuries
            },
            "actions": {
                "planned": self.actions_planned,
                "executed": self.actions_executed,
                "successful": self.actions_successful,
                "execution_rate": self.actions_executed / self.actions_planned if self.actions_planned > 0 else 0,
                "success_rate": self.actions_successful / self.actions_executed if self.actions_executed > 0 else 0
            },
            "evacuation": {
                "ordered": self.evacuation_ordered,
                "rate": self.evacuation_rate,
                "shelter_utilization": self.shelter_utilization
            },
            "timing": {
                "lead_time_hours": self.lead_time_hours,
                "lead_time_used_hours": self.lead_time_used_hours,
                "lead_time_efficiency": self.lead_time_used_hours / self.lead_time_hours if self.lead_time_hours > 0 else 0
            },
            "assessment": {
                "false_alarm": self.false_alarm,
                "missed_event": self.missed_event,
                "confidence_rating": self.confidence_rating
            },
            "notes": self.notes,
            "recorded_by": self.recorded_by,
            "recorded_at": self.recorded_at
        }


@dataclass
class OutcomeAnalysis:
    """Analysis of a prediction outcome"""
    prediction_accurate: bool
    severity_accurate: bool
    timing_accurate: bool
    lead_time_sufficient: bool
    actions_effective: bool
    evacuation_adequate: bool
    lives_protected: bool
    overall_score: float
    improvements_needed: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "prediction_accurate": self.prediction_accurate,
            "severity_accurate": self.severity_accurate,
            "timing_accurate": self.timing_accurate,
            "lead_time_sufficient": self.lead_time_sufficient,
            "actions_effective": self.actions_effective,
            "evacuation_adequate": self.evacuation_adequate,
            "lives_protected": self.lives_protected,
            "overall_score": self.overall_score,
            "improvements_needed": self.improvements_needed
        }


@dataclass
class SystemMetrics:
    """Overall system performance metrics"""
    total_predictions: int
    total_disasters: int
    true_positives: int        # Correctly predicted disasters
    true_negatives: int        # Correctly predicted no disaster
    false_positives: int       # False alarms
    false_negatives: int       # Missed events
    
    # Calculated metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_alarm_rate: float
    miss_rate: float
    
    # Timing metrics
    average_lead_time_hours: float
    average_lead_time_used_hours: float
    
    # Action metrics
    average_action_execution_rate: float
    average_action_success_rate: float
    average_evacuation_rate: float
    
    # Impact metrics
    total_lives_saved_estimate: int
    total_damage_prevented_estimate: float
    
    # Trend
    accuracy_trend: str  # "improving", "stable", "declining"
    
    def to_dict(self) -> Dict:
        return {
            "counts": {
                "total_predictions": self.total_predictions,
                "total_disasters": self.total_disasters,
                "true_positives": self.true_positives,
                "true_negatives": self.true_negatives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives
            },
            "accuracy_metrics": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "false_alarm_rate": self.false_alarm_rate,
                "miss_rate": self.miss_rate
            },
            "timing_metrics": {
                "average_lead_time_hours": self.average_lead_time_hours,
                "average_lead_time_used_hours": self.average_lead_time_used_hours
            },
            "action_metrics": {
                "average_action_execution_rate": self.average_action_execution_rate,
                "average_action_success_rate": self.average_action_success_rate,
                "average_evacuation_rate": self.average_evacuation_rate
            },
            "impact_metrics": {
                "total_lives_saved_estimate": self.total_lives_saved_estimate,
                "total_damage_prevented_estimate": self.total_damage_prevented_estimate
            },
            "trend": self.accuracy_trend
        }


class FeedbackSystem:
    """
    Collect outcomes and use them to improve models
    
    Implements virtuous cycle:
    1. Make prediction → stored with prediction_id
    2. Alert sent → stored with alert_id
    3. Actions taken → tracked
    4. Event occurs (or doesn't) → recorded
    5. Outcome analyzed → improvements identified
    6. Model updated → better predictions
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.outcomes_file = self.data_dir / "prediction_outcomes.jsonl"
        self.metrics_file = self.data_dir / "metrics" / "feedback_metrics.json"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "metrics").mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of recent outcomes
        self.outcomes_cache: List[PredictionOutcome] = []
        self.max_cache_size = 1000
        
        # Load existing outcomes
        self._load_outcomes()
        
        logger.info(f"FeedbackSystem initialized with {len(self.outcomes_cache)} cached outcomes")
    
    def _load_outcomes(self):
        """Load outcomes from file into cache"""
        if self.outcomes_file.exists():
            try:
                with open(self.outcomes_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # Convert dict back to PredictionOutcome
                            outcome = self._dict_to_outcome(data.get('outcome', data))
                            self.outcomes_cache.append(outcome)
                
                # Keep only most recent
                if len(self.outcomes_cache) > self.max_cache_size:
                    self.outcomes_cache = self.outcomes_cache[-self.max_cache_size:]
                    
            except Exception as e:
                logger.error(f"Failed to load outcomes: {e}")
    
    def _dict_to_outcome(self, data: Dict) -> PredictionOutcome:
        """Convert dictionary to PredictionOutcome"""
        # Handle nested structure
        predicted = data.get('predicted', {})
        actual = data.get('actual', {})
        actions = data.get('actions', {})
        evacuation = data.get('evacuation', {})
        timing = data.get('timing', {})
        assessment = data.get('assessment', {})
        
        return PredictionOutcome(
            prediction_id=data.get('prediction_id', ''),
            alert_id=data.get('alert_id', ''),
            location=data.get('location', ''),
            hazard_type=data.get('hazard_type', ''),
            prediction_timestamp=data.get('prediction_timestamp', ''),
            predicted_risk_level=predicted.get('risk_level', data.get('predicted_risk_level', '')),
            predicted_risk_score=predicted.get('risk_score', data.get('predicted_risk_score', 0)),
            predicted_peak_hours=predicted.get('peak_hours', data.get('predicted_peak_hours', 0)),
            predicted_severity=predicted.get('severity', data.get('predicted_severity', '')),
            method_breakdown=predicted.get('method_breakdown', data.get('method_breakdown', {})),
            actual_disaster_occurred=actual.get('disaster_occurred', data.get('actual_disaster_occurred', False)),
            actual_peak_time=actual.get('peak_time', data.get('actual_peak_time')),
            actual_severity=actual.get('severity', data.get('actual_severity')),
            actual_damage_estimate=actual.get('damage_estimate', data.get('actual_damage_estimate', 0)),
            actual_affected_population=actual.get('affected_population', data.get('actual_affected_population', 0)),
            actual_casualties=actual.get('casualties', data.get('actual_casualties', 0)),
            actual_injuries=actual.get('injuries', data.get('actual_injuries', 0)),
            actions_planned=actions.get('planned', data.get('actions_planned', 0)),
            actions_executed=actions.get('executed', data.get('actions_executed', 0)),
            actions_successful=actions.get('successful', data.get('actions_successful', 0)),
            evacuation_ordered=evacuation.get('ordered', data.get('evacuation_ordered', False)),
            evacuation_rate=evacuation.get('rate', data.get('evacuation_rate', 0)),
            shelter_utilization=evacuation.get('shelter_utilization', data.get('shelter_utilization', 0)),
            lead_time_hours=timing.get('lead_time_hours', data.get('lead_time_hours', 0)),
            lead_time_used_hours=timing.get('lead_time_used_hours', data.get('lead_time_used_hours', 0)),
            false_alarm=assessment.get('false_alarm', data.get('false_alarm', False)),
            missed_event=assessment.get('missed_event', data.get('missed_event', False)),
            confidence_rating=assessment.get('confidence_rating', data.get('confidence_rating', 0)),
            notes=data.get('notes', ''),
            recorded_by=data.get('recorded_by', ''),
            recorded_at=data.get('recorded_at', datetime.utcnow().isoformat())
        )
    
    async def record_outcome(self, outcome: PredictionOutcome) -> Dict:
        """
        Record what actually happened after a prediction
        
        Called post-disaster by emergency manager or automated monitoring
        
        Args:
            outcome: PredictionOutcome with actual results
            
        Returns:
            Dict with analysis and improvement suggestions
        """
        # Analyze the outcome
        analysis = self._analyze_outcome(outcome)
        
        # Add to cache
        self.outcomes_cache.append(outcome)
        if len(self.outcomes_cache) > self.max_cache_size:
            self.outcomes_cache = self.outcomes_cache[-self.max_cache_size:]
        
        # Persist to file
        self._save_outcome(outcome, analysis)
        
        # Identify improvements
        improvements = self._identify_improvements(outcome, analysis)
        
        # Update metrics
        await self._update_metrics()
        
        logger.info(
            f"Outcome recorded for {outcome.prediction_id}: "
            f"accurate={analysis.prediction_accurate}, "
            f"score={analysis.overall_score:.2f}"
        )
        
        return {
            "outcome_id": outcome.prediction_id,
            "recorded": True,
            "analysis": analysis.to_dict(),
            "improvements": improvements,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _analyze_outcome(self, outcome: PredictionOutcome) -> OutcomeAnalysis:
        """Analyze how accurate the prediction was"""
        improvements = []
        
        # Prediction accuracy
        prediction_accurate = not outcome.false_alarm and not outcome.missed_event
        if outcome.false_alarm:
            improvements.append("Reduce false positive rate - model overestimated risk")
        if outcome.missed_event:
            improvements.append("Improve detection - disaster was not predicted")
        
        # Severity accuracy
        severity_accurate = False
        if outcome.actual_disaster_occurred and outcome.actual_severity:
            predicted_sev = outcome.predicted_severity
            actual_sev = outcome.actual_severity
            # Consider accurate if within one level
            severity_levels = ["Minor", "Moderate", "Severe", "Extreme"]
            if predicted_sev in severity_levels and actual_sev in severity_levels:
                pred_idx = severity_levels.index(predicted_sev)
                act_idx = severity_levels.index(actual_sev)
                severity_accurate = abs(pred_idx - act_idx) <= 1
                if not severity_accurate:
                    improvements.append(f"Improve severity estimation - predicted {predicted_sev}, actual {actual_sev}")
        elif not outcome.actual_disaster_occurred:
            severity_accurate = True  # N/A
        
        # Timing accuracy
        timing_accurate = True
        if outcome.actual_disaster_occurred and outcome.actual_peak_time:
            # Compare predicted vs actual peak time
            # For now, consider accurate if within 6 hours
            timing_accurate = True  # Simplified - would need actual calculation
        
        # Lead time sufficient
        lead_time_sufficient = outcome.lead_time_hours >= 24
        if not lead_time_sufficient:
            improvements.append(f"Increase lead time - only {outcome.lead_time_hours}h warning provided")
        
        # Actions effective
        action_execution_rate = outcome.actions_executed / outcome.actions_planned if outcome.actions_planned > 0 else 0
        action_success_rate = outcome.actions_successful / outcome.actions_executed if outcome.actions_executed > 0 else 0
        actions_effective = action_execution_rate >= 0.8 and action_success_rate >= 0.8
        if action_execution_rate < 0.8:
            improvements.append(f"Improve action execution - only {action_execution_rate:.0%} of actions completed")
        if action_success_rate < 0.8:
            improvements.append(f"Improve action success - only {action_success_rate:.0%} of actions succeeded")
        
        # Evacuation adequate
        evacuation_adequate = True
        if outcome.evacuation_ordered:
            evacuation_adequate = outcome.evacuation_rate >= 0.90
            if not evacuation_adequate:
                improvements.append(f"Improve evacuation rate - only {outcome.evacuation_rate:.0%} evacuated")
        
        # Lives protected
        lives_protected = outcome.actual_casualties == 0
        if not lives_protected:
            improvements.append(f"Critical: {outcome.actual_casualties} casualties occurred")
        
        # Calculate overall score
        scores = [
            1.0 if prediction_accurate else 0.0,
            1.0 if severity_accurate else 0.5,
            1.0 if timing_accurate else 0.5,
            1.0 if lead_time_sufficient else 0.5,
            action_execution_rate,
            action_success_rate,
            1.0 if evacuation_adequate else outcome.evacuation_rate,
            1.0 if lives_protected else 0.0
        ]
        overall_score = sum(scores) / len(scores)
        
        return OutcomeAnalysis(
            prediction_accurate=prediction_accurate,
            severity_accurate=severity_accurate,
            timing_accurate=timing_accurate,
            lead_time_sufficient=lead_time_sufficient,
            actions_effective=actions_effective,
            evacuation_adequate=evacuation_adequate,
            lives_protected=lives_protected,
            overall_score=overall_score,
            improvements_needed=improvements
        )
    
    def _identify_improvements(self, outcome: PredictionOutcome, analysis: OutcomeAnalysis) -> List[str]:
        """Identify systematic improvements based on outcome"""
        improvements = list(analysis.improvements_needed)
        
        # Add method-specific improvements
        if outcome.method_breakdown:
            # If one method was significantly off, suggest retraining
            for method, score in outcome.method_breakdown.items():
                if outcome.actual_disaster_occurred and score < 0.3:
                    improvements.append(f"Retrain {method} model - underestimated risk")
                elif not outcome.actual_disaster_occurred and score > 0.7:
                    improvements.append(f"Adjust {method} model - overestimated risk")
        
        # Confidence calibration
        if outcome.confidence_rating < 0.5:
            improvements.append("Improve alert clarity - emergency managers had low confidence in prediction")
        
        return improvements
    
    def _save_outcome(self, outcome: PredictionOutcome, analysis: OutcomeAnalysis):
        """Save outcome to persistent storage"""
        try:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "outcome": outcome.to_dict(),
                "analysis": analysis.to_dict()
            }
            
            with open(self.outcomes_file, 'a') as f:
                f.write(json.dumps(record) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to save outcome: {e}")
    
    async def _update_metrics(self):
        """Update system-wide metrics"""
        try:
            metrics = self.calculate_system_metrics()
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def calculate_system_metrics(self) -> SystemMetrics:
        """Calculate overall system accuracy and performance metrics"""
        if not self.outcomes_cache:
            return SystemMetrics(
                total_predictions=0,
                total_disasters=0,
                true_positives=0,
                true_negatives=0,
                false_positives=0,
                false_negatives=0,
                accuracy=0,
                precision=0,
                recall=0,
                f1_score=0,
                false_alarm_rate=0,
                miss_rate=0,
                average_lead_time_hours=0,
                average_lead_time_used_hours=0,
                average_action_execution_rate=0,
                average_action_success_rate=0,
                average_evacuation_rate=0,
                total_lives_saved_estimate=0,
                total_damage_prevented_estimate=0,
                accuracy_trend="stable"
            )
        
        total = len(self.outcomes_cache)
        
        # Count outcomes
        true_positives = sum(1 for o in self.outcomes_cache 
                           if o.actual_disaster_occurred and not o.missed_event and o.predicted_risk_level in ["HIGH", "CRITICAL"])
        true_negatives = sum(1 for o in self.outcomes_cache 
                            if not o.actual_disaster_occurred and not o.false_alarm)
        false_positives = sum(1 for o in self.outcomes_cache if o.false_alarm)
        false_negatives = sum(1 for o in self.outcomes_cache if o.missed_event)
        
        total_disasters = sum(1 for o in self.outcomes_cache if o.actual_disaster_occurred)
        
        # Calculate metrics
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        false_alarm_rate = false_positives / total if total > 0 else 0
        miss_rate = false_negatives / total_disasters if total_disasters > 0 else 0
        
        # Timing metrics
        lead_times = [o.lead_time_hours for o in self.outcomes_cache if o.lead_time_hours > 0]
        avg_lead_time = sum(lead_times) / len(lead_times) if lead_times else 0
        
        lead_times_used = [o.lead_time_used_hours for o in self.outcomes_cache if o.lead_time_used_hours > 0]
        avg_lead_time_used = sum(lead_times_used) / len(lead_times_used) if lead_times_used else 0
        
        # Action metrics
        action_execution_rates = [
            o.actions_executed / o.actions_planned 
            for o in self.outcomes_cache 
            if o.actions_planned > 0
        ]
        avg_action_execution = sum(action_execution_rates) / len(action_execution_rates) if action_execution_rates else 0
        
        action_success_rates = [
            o.actions_successful / o.actions_executed 
            for o in self.outcomes_cache 
            if o.actions_executed > 0
        ]
        avg_action_success = sum(action_success_rates) / len(action_success_rates) if action_success_rates else 0
        
        evacuation_rates = [o.evacuation_rate for o in self.outcomes_cache if o.evacuation_ordered]
        avg_evacuation = sum(evacuation_rates) / len(evacuation_rates) if evacuation_rates else 0
        
        # Impact estimates
        # Estimate lives saved based on successful evacuations
        lives_saved = sum(
            int(o.actual_affected_population * o.evacuation_rate * 0.01)  # 1% of evacuated would have been casualties
            for o in self.outcomes_cache 
            if o.actual_disaster_occurred and o.evacuation_ordered
        )
        
        # Estimate damage prevented (very rough)
        damage_prevented = sum(
            o.actual_damage_estimate * 0.3  # Assume 30% reduction from early warning
            for o in self.outcomes_cache 
            if o.actual_disaster_occurred and not o.missed_event
        )
        
        # Trend analysis (compare recent vs older)
        if len(self.outcomes_cache) >= 20:
            recent = self.outcomes_cache[-10:]
            older = self.outcomes_cache[-20:-10]
            
            recent_accuracy = sum(1 for o in recent if not o.false_alarm and not o.missed_event) / len(recent)
            older_accuracy = sum(1 for o in older if not o.false_alarm and not o.missed_event) / len(older)
            
            if recent_accuracy > older_accuracy + 0.05:
                trend = "improving"
            elif recent_accuracy < older_accuracy - 0.05:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return SystemMetrics(
            total_predictions=total,
            total_disasters=total_disasters,
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_alarm_rate=false_alarm_rate,
            miss_rate=miss_rate,
            average_lead_time_hours=avg_lead_time,
            average_lead_time_used_hours=avg_lead_time_used,
            average_action_execution_rate=avg_action_execution,
            average_action_success_rate=avg_action_success,
            average_evacuation_rate=avg_evacuation,
            total_lives_saved_estimate=lives_saved,
            total_damage_prevented_estimate=damage_prevented,
            accuracy_trend=trend
        )
    
    def identify_pattern_improvements(self) -> List[str]:
        """Identify systematic improvements from outcome patterns"""
        improvements = []
        metrics = self.calculate_system_metrics()
        
        # Check false alarm rate
        if metrics.false_alarm_rate > 0.20:
            improvements.append(
                f"False alarm rate too high ({metrics.false_alarm_rate:.0%}). "
                "Consider: Raise risk thresholds, retrain with better negative examples, "
                "improve pattern matching specificity."
            )
        
        # Check miss rate
        if metrics.miss_rate > 0.10:
            improvements.append(
                f"Miss rate concerning ({metrics.miss_rate:.0%}). "
                "Consider: Lower detection thresholds, add more data sources, "
                "improve early warning indicators."
            )
        
        # Check lead time
        if metrics.average_lead_time_hours < 24:
            improvements.append(
                f"Average lead time insufficient ({metrics.average_lead_time_hours:.1f}h). "
                "Consider: Extend forecast horizon, improve early detection algorithms."
            )
        
        # Check action execution
        if metrics.average_action_execution_rate < 0.80:
            improvements.append(
                f"Action execution rate low ({metrics.average_action_execution_rate:.0%}). "
                "Consider: Simplify action plans, improve stakeholder training, "
                "better resource pre-positioning."
            )
        
        # Check evacuation
        if metrics.average_evacuation_rate < 0.90:
            improvements.append(
                f"Evacuation rate below target ({metrics.average_evacuation_rate:.0%}). "
                "Consider: Improve alert distribution, clearer instructions, "
                "better evacuation route planning."
            )
        
        # Analyze by hazard type
        hazard_performance = self._analyze_by_hazard_type()
        for hazard, perf in hazard_performance.items():
            if perf['accuracy'] < 0.70:
                improvements.append(
                    f"{hazard.title()} prediction accuracy low ({perf['accuracy']:.0%}). "
                    f"Consider: More {hazard}-specific training data, specialized models."
                )
        
        return improvements
    
    def _analyze_by_hazard_type(self) -> Dict[str, Dict]:
        """Analyze performance by hazard type"""
        hazard_outcomes = {}
        
        for outcome in self.outcomes_cache:
            hazard = outcome.hazard_type
            if hazard not in hazard_outcomes:
                hazard_outcomes[hazard] = []
            hazard_outcomes[hazard].append(outcome)
        
        performance = {}
        for hazard, outcomes in hazard_outcomes.items():
            total = len(outcomes)
            accurate = sum(1 for o in outcomes if not o.false_alarm and not o.missed_event)
            performance[hazard] = {
                "total": total,
                "accurate": accurate,
                "accuracy": accurate / total if total > 0 else 0
            }
        
        return performance
    
    def get_recent_outcomes(self, limit: int = 10) -> List[Dict]:
        """Get most recent outcomes"""
        recent = self.outcomes_cache[-limit:]
        return [o.to_dict() for o in reversed(recent)]
    
    def get_outcome_by_prediction(self, prediction_id: str) -> Optional[PredictionOutcome]:
        """Get outcome for a specific prediction"""
        for outcome in self.outcomes_cache:
            if outcome.prediction_id == prediction_id:
                return outcome
        return None
    
    def generate_feedback_report(self, days: int = 30) -> Dict:
        """Generate comprehensive feedback report"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        recent_outcomes = [
            o for o in self.outcomes_cache 
            if o.recorded_at and datetime.fromisoformat(o.recorded_at.replace('Z', '+00:00').replace('+00:00', '')) > cutoff
        ]
        
        metrics = self.calculate_system_metrics()
        improvements = self.identify_pattern_improvements()
        
        return {
            "report_period_days": days,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_outcomes_recorded": len(recent_outcomes),
                "overall_accuracy": metrics.accuracy,
                "false_alarm_rate": metrics.false_alarm_rate,
                "miss_rate": metrics.miss_rate,
                "average_lead_time": metrics.average_lead_time_hours,
                "lives_saved_estimate": metrics.total_lives_saved_estimate,
                "trend": metrics.accuracy_trend
            },
            "detailed_metrics": metrics.to_dict(),
            "recommended_improvements": improvements,
            "hazard_breakdown": self._analyze_by_hazard_type()
        }


# Factory function
def create_feedback_system(data_dir: str = "data") -> FeedbackSystem:
    """Create a FeedbackSystem instance"""
    return FeedbackSystem(data_dir=data_dir)
