"""
Forecast Agent

Synthesizes predictions from multiple methods:
1. LSTM model predictions
2. Pattern matching against historical disasters
3. Progression analysis trajectory
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("crisisconnect.agents.forecast")


@dataclass
class Forecast:
    """Prediction about future conditions"""
    hazard_type: str
    risk_score: float  # 0-1
    hours_to_peak: int
    confidence: float
    method: str  # "lstm", "pattern_matching", "progression"
    reasoning: str  # Human-readable explanation
    details: Dict[str, Any] = None  # Additional method-specific details
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "hazard_type": self.hazard_type,
            "risk_score": self.risk_score,
            "hours_to_peak": self.hours_to_peak,
            "confidence": self.confidence,
            "method": self.method,
            "reasoning": self.reasoning,
            "details": self.details or {}
        }


class ForecastAgent:
    """
    Agent that synthesizes multiple prediction approaches
    
    Methods:
    1. Pattern matching: "This looks like the 2023 flood pattern"
    2. LSTM: "Sequence predicts 65% flood risk"
    3. Progression: "Severity is 0.8 and worsening at 0.2/day = peak in 1 day"
    
    Each method provides a forecast with confidence, which the ensemble
    coordinator will combine.
    """
    
    def __init__(self, 
                 lstm_model=None, 
                 pattern_agent=None, 
                 progression_agent=None):
        """
        Initialize forecast agent with optional sub-agents/models
        
        Args:
            lstm_model: Trained LSTM model for sequence prediction
            pattern_agent: PatternDetectionAgent instance
            progression_agent: ProgressionAnalyzerAgent instance
        """
        self.lstm_model = lstm_model
        self.pattern_agent = pattern_agent
        self.progression_agent = progression_agent
        
        logger.info(f"ForecastAgent initialized: lstm={lstm_model is not None}, "
                   f"pattern={pattern_agent is not None}, progression={progression_agent is not None}")
    
    def set_lstm_model(self, model):
        """Set or update the LSTM model"""
        self.lstm_model = model
        logger.info("LSTM model updated")
    
    def set_pattern_agent(self, agent):
        """Set or update the pattern detection agent"""
        self.pattern_agent = agent
        logger.info("Pattern agent updated")
    
    def set_progression_agent(self, agent):
        """Set or update the progression analyzer agent"""
        self.progression_agent = agent
        logger.info("Progression agent updated")
    
    def forecast(self, 
                current_sequence: np.ndarray,
                weather_history: pd.DataFrame,
                hazard_type: str = "flood",
                feature_columns: List[str] = None) -> List[Forecast]:
        """
        Generate forecasts from all available methods
        
        Args:
            current_sequence: Current weather sequence for LSTM (shape: timesteps x features)
            weather_history: DataFrame with weather data for progression analysis
            hazard_type: Type of hazard to forecast
            feature_columns: Feature column names for pattern matching
            
        Returns:
            List of Forecast objects from each method
        """
        forecasts = []
        
        # Method 1: LSTM prediction
        if self.lstm_model is not None:
            lstm_forecast = self._lstm_forecast(current_sequence, hazard_type)
            if lstm_forecast:
                forecasts.append(lstm_forecast)
        
        # Method 2: Pattern matching
        if self.pattern_agent is not None:
            pattern_forecast = self._pattern_forecast(current_sequence, hazard_type, feature_columns)
            if pattern_forecast:
                forecasts.append(pattern_forecast)
        
        # Method 3: Progression analysis
        if self.progression_agent is not None:
            progression_forecast = self._progression_forecast(weather_history, hazard_type)
            if progression_forecast:
                forecasts.append(progression_forecast)
        
        if not forecasts:
            # Fallback: simple heuristic forecast
            fallback = self._fallback_forecast(weather_history, hazard_type)
            forecasts.append(fallback)
        
        return forecasts
    
    def _lstm_forecast(self, sequence: np.ndarray, hazard_type: str) -> Optional[Forecast]:
        """
        Get LSTM model prediction
        
        Args:
            sequence: Input sequence (timesteps x features or batch x timesteps x features)
            hazard_type: Type of hazard
            
        Returns:
            Forecast object or None if prediction fails
        """
        try:
            # Ensure correct shape
            if sequence.ndim == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            elif sequence.ndim == 1:
                # Can't use 1D sequence for LSTM
                logger.warning("LSTM requires 2D or 3D sequence input")
                return None
            
            # Get prediction
            predictions = self.lstm_model.predict(sequence)
            
            # Handle different output formats
            if isinstance(predictions, tuple):
                risk_scores, confidence_scores = predictions
                risk_score = float(risk_scores[0]) if len(risk_scores) > 0 else 0.5
                confidence = float(confidence_scores[0]) if len(confidence_scores) > 0 else 0.5
            elif isinstance(predictions, np.ndarray):
                if predictions.ndim > 1:
                    # Multi-horizon output
                    risk_score = float(predictions[0].mean())  # Average across horizons
                else:
                    risk_score = float(predictions[0])
                confidence = 0.7  # Default confidence for LSTM
            else:
                risk_score = float(predictions)
                confidence = 0.7
            
            # Ensure risk_score is in 0-1 range
            if risk_score > 1:
                risk_score = risk_score / 100  # Might be percentage
            
            hours_to_peak = self._estimate_hours_from_risk(risk_score)
            
            return Forecast(
                hazard_type=hazard_type,
                risk_score=risk_score,
                hours_to_peak=hours_to_peak,
                confidence=confidence,
                method="lstm",
                reasoning=f"LSTM model predicts {risk_score:.1%} {hazard_type} risk based on {sequence.shape[1]}-step sequence",
                details={
                    "sequence_length": sequence.shape[1],
                    "raw_prediction": float(risk_score)
                }
            )
            
        except Exception as e:
            logger.warning(f"LSTM forecast failed: {e}")
            return None
    
    def _pattern_forecast(self, sequence: np.ndarray, hazard_type: str, 
                         feature_columns: List[str] = None) -> Optional[Forecast]:
        """
        Get pattern matching prediction
        
        Args:
            sequence: Current weather sequence
            hazard_type: Type of hazard
            feature_columns: Feature column names
            
        Returns:
            Forecast object or None if no patterns match
        """
        try:
            # Flatten sequence if needed for pattern matching
            if sequence.ndim == 3:
                # Take mean across timesteps or flatten
                flat_sequence = sequence.mean(axis=1)[0]  # Average across time
            elif sequence.ndim == 2:
                flat_sequence = sequence.mean(axis=0)  # Average across time
            else:
                flat_sequence = sequence
            
            matches = self.pattern_agent.match_current_sequence(flat_sequence)
            
            if not matches:
                return None
            
            best_match, similarity = matches[0]
            
            # Risk score based on pattern reliability and similarity
            risk_score = similarity * best_match.precedes_disaster_pct
            hours_to_peak = int(best_match.days_to_disaster_avg * 24)
            
            return Forecast(
                hazard_type=hazard_type,
                risk_score=risk_score,
                hours_to_peak=hours_to_peak,
                confidence=similarity,
                method="pattern_matching",
                reasoning=(
                    f"Matches historical pattern '{best_match.pattern_id}' "
                    f"({best_match.frequency} occurrences, {best_match.precedes_disaster_pct:.1%} led to {hazard_type})"
                ),
                details={
                    "pattern_id": best_match.pattern_id,
                    "pattern_frequency": best_match.frequency,
                    "similarity": similarity,
                    "pattern_reliability": best_match.precedes_disaster_pct,
                    "top_features": dict(list(best_match.feature_importance.items())[:5])
                }
            )
            
        except Exception as e:
            logger.warning(f"Pattern forecast failed: {e}")
            return None
    
    def _progression_forecast(self, weather_data: pd.DataFrame, 
                             hazard_type: str) -> Optional[Forecast]:
        """
        Get progression analysis prediction
        
        Args:
            weather_data: DataFrame with weather history
            hazard_type: Type of hazard
            
        Returns:
            Forecast object or None if analysis fails
        """
        try:
            analysis = self.progression_agent.analyze_progression(weather_data, hazard_type)
            
            # Map stage to risk score
            stage_scores = {
                0: 0.15,  # DORMANT
                1: 0.45,  # ESCALATING
                2: 0.75,  # CRITICAL
                3: 0.95   # PEAK
            }
            
            base_risk = stage_scores.get(analysis.current_stage.value, 0.5)
            
            # Adjust based on velocity and acceleration
            velocity_adjustment = analysis.worsening_velocity * 0.1
            accel_adjustment = 0.05 if analysis.is_accelerating else 0
            
            risk_score = min(1.0, max(0.0, base_risk + velocity_adjustment + accel_adjustment))
            hours_to_peak = analysis.days_to_peak_estimate * 24
            
            return Forecast(
                hazard_type=hazard_type,
                risk_score=risk_score,
                hours_to_peak=int(hours_to_peak),
                confidence=analysis.confidence,
                method="progression",
                reasoning=(
                    f"Conditions in {analysis.current_stage.name} stage "
                    f"(severity {analysis.severity_score:.1%}), "
                    f"{'worsening' if analysis.worsening_velocity > 0 else 'improving'} "
                    f"at {abs(analysis.worsening_velocity):.1%}/day"
                ),
                details={
                    "stage": analysis.current_stage.name,
                    "severity": analysis.severity_score,
                    "velocity": analysis.worsening_velocity,
                    "is_accelerating": analysis.is_accelerating,
                    "key_indicators": analysis.key_indicators,
                    "stage_history": analysis.stage_history
                }
            )
            
        except Exception as e:
            logger.warning(f"Progression forecast failed: {e}")
            return None
    
    def _fallback_forecast(self, weather_data: pd.DataFrame, 
                          hazard_type: str) -> Forecast:
        """
        Generate simple heuristic forecast when no agents are available
        
        Args:
            weather_data: DataFrame with weather data
            hazard_type: Type of hazard
            
        Returns:
            Basic Forecast object
        """
        risk_score = 0.3  # Default moderate-low risk
        
        # Simple heuristics based on available data
        if weather_data is not None and len(weather_data) > 0:
            # Check for high rainfall
            for col in ['rainfall', 'precip_mm', 'precipitation']:
                if col in weather_data.columns:
                    max_rain = weather_data[col].max()
                    if max_rain > 50:
                        risk_score = max(risk_score, 0.6)
                    elif max_rain > 20:
                        risk_score = max(risk_score, 0.4)
                    break
            
            # Check for high wind
            for col in ['wind_speed', 'wind_kph']:
                if col in weather_data.columns:
                    max_wind = weather_data[col].max()
                    if max_wind > 60:
                        risk_score = max(risk_score, 0.5)
                    break
        
        return Forecast(
            hazard_type=hazard_type,
            risk_score=risk_score,
            hours_to_peak=48,
            confidence=0.3,  # Low confidence for fallback
            method="heuristic",
            reasoning="Basic heuristic assessment (no trained models available)",
            details={"note": "Consider training models for better predictions"}
        )
    
    def _estimate_hours_from_risk(self, risk_score: float) -> int:
        """
        Estimate hours to peak based on risk score
        
        Higher risk = sooner peak (conditions already severe)
        """
        if risk_score < 0.3:
            return 72  # 3 days
        elif risk_score < 0.5:
            return 48  # 2 days
        elif risk_score < 0.7:
            return 24  # 1 day
        elif risk_score < 0.85:
            return 12  # 12 hours
        else:
            return 6   # 6 hours (imminent)
    
    def get_forecast_summary(self, forecasts: List[Forecast]) -> Dict:
        """
        Generate summary of all forecasts
        
        Args:
            forecasts: List of Forecast objects
            
        Returns:
            Summary dictionary
        """
        if not forecasts:
            return {"error": "No forecasts available"}
        
        risk_scores = [f.risk_score for f in forecasts]
        confidences = [f.confidence for f in forecasts]
        
        return {
            "num_forecasts": len(forecasts),
            "methods_used": [f.method for f in forecasts],
            "risk_scores": {
                "min": min(risk_scores),
                "max": max(risk_scores),
                "mean": np.mean(risk_scores),
                "std": np.std(risk_scores)
            },
            "avg_confidence": np.mean(confidences),
            "forecasts": [f.to_dict() for f in forecasts]
        }
