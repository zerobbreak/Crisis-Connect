"""
Progression Analyzer Agent

Tracks how conditions evolve from "safe" to "dangerous" state,
analyzing the trajectory of weather conditions toward disaster.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("crisisconnect.agents.progression_analyzer")


class ProgressionStage(Enum):
    """Stages of disaster development"""
    DORMANT = 0      # Normal conditions
    ESCALATING = 1   # Conditions worsening
    CRITICAL = 2     # High risk
    PEAK = 3         # Maximum danger


@dataclass
class ProgressionAnalysis:
    """Result of analyzing current conditions"""
    current_stage: ProgressionStage
    severity_score: float  # 0-1
    worsening_velocity: float  # Rate of change (-1 to 1)
    is_accelerating: bool  # Is it getting worse faster?
    days_to_peak_estimate: int
    confidence: float
    key_indicators: Dict[str, float]  # Which indicators are concerning
    stage_history: List[str]  # Recent stage transitions


class ProgressionAnalyzerAgent:
    """
    Agent that analyzes how conditions are developing toward disaster
    
    Key insight: Just because conditions are bad doesn't mean disaster is imminent.
    What matters is the TRAJECTORY:
    - Stable bad conditions: Lower risk
    - Rapidly worsening conditions: HIGH risk
    - Accelerating deterioration: CRITICAL risk
    """
    
    def __init__(self):
        """Initialize progression analyzer"""
        self.stage_thresholds = {
            ProgressionStage.DORMANT: 0.25,
            ProgressionStage.ESCALATING: 0.50,
            ProgressionStage.CRITICAL: 0.75
        }
        self.stage_history: List[ProgressionStage] = []
        
        # Hazard-specific indicators
        self.hazard_indicators = {
            "flood": {
                "primary": ["rainfall", "precip_mm", "precipitation"],
                "secondary": ["soil_moisture", "humidity", "wind_speed", "wind_kph"],
                "thresholds": {
                    "rainfall_high": 50,  # mm
                    "rainfall_extreme": 100,
                    "humidity_high": 85,
                    "soil_moisture_high": 80
                }
            },
            "drought": {
                "primary": ["rainfall", "precip_mm", "precipitation", "temperature", "temp_c"],
                "secondary": ["soil_moisture", "humidity"],
                "thresholds": {
                    "rainfall_low": 10,  # mm cumulative
                    "temperature_high": 35,
                    "humidity_low": 30,
                    "soil_moisture_low": 20
                }
            },
            "storm": {
                "primary": ["wind_speed", "wind_kph", "pressure_mb", "pressure"],
                "secondary": ["rainfall", "precip_mm"],
                "thresholds": {
                    "wind_high": 60,  # kph
                    "wind_extreme": 100,
                    "pressure_low": 1000  # mb
                }
            }
        }
        
        logger.info("ProgressionAnalyzerAgent initialized")
    
    def analyze_progression(self, 
                           weather_history: pd.DataFrame,
                           hazard_type: str = "flood",
                           lookback_days: int = 14) -> ProgressionAnalysis:
        """
        Analyze how conditions are progressing toward disaster
        
        Input: Weather data (ideally 14+ days)
        Output: Which stage are we in? How fast is it worsening? When will it peak?
        
        Args:
            weather_history: DataFrame with weather data
            hazard_type: Type of hazard to analyze ("flood", "drought", "storm")
            lookback_days: Number of days to analyze
            
        Returns:
            ProgressionAnalysis with current assessment
        """
        # Normalize hazard type
        hazard_type = hazard_type.lower()
        if hazard_type not in self.hazard_indicators:
            hazard_type = "flood"  # Default
        
        # Get recent data
        if len(weather_history) > lookback_days:
            recent = weather_history.tail(lookback_days).copy()
        else:
            recent = weather_history.copy()
        
        if len(recent) < 3:
            return ProgressionAnalysis(
                current_stage=ProgressionStage.DORMANT,
                severity_score=0.0,
                worsening_velocity=0.0,
                is_accelerating=False,
                days_to_peak_estimate=999,
                confidence=0.1,
                key_indicators={},
                stage_history=[]
            )
        
        # Calculate severity indicators based on hazard type
        severity, key_indicators = self._calculate_severity(recent, hazard_type)
        
        # Calculate velocity (how fast is it changing?)
        velocity = self._calculate_velocity(recent, hazard_type)
        
        # Is it accelerating?
        acceleration = self._detect_acceleration(recent, hazard_type)
        
        # Classify current stage
        current_stage = self._classify_stage(severity, velocity)
        
        # Track stage history
        self.stage_history.append(current_stage)
        if len(self.stage_history) > 10:
            self.stage_history = self.stage_history[-10:]
        
        # Estimate days to peak
        days_to_peak = self._estimate_days_to_peak(severity, velocity, acceleration, hazard_type)
        
        # Confidence in analysis
        data_completeness = min(len(recent) / lookback_days, 1.0)
        velocity_certainty = 1 - abs(velocity) / 2  # More stable = more certain
        confidence = data_completeness * 0.6 + velocity_certainty * 0.4
        
        return ProgressionAnalysis(
            current_stage=current_stage,
            severity_score=severity,
            worsening_velocity=velocity,
            is_accelerating=acceleration > 0,
            days_to_peak_estimate=days_to_peak,
            confidence=confidence,
            key_indicators=key_indicators,
            stage_history=[s.name for s in self.stage_history[-5:]]
        )
    
    def _calculate_severity(self, weather_data: pd.DataFrame, hazard_type: str) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite severity score 0-1 based on hazard type
        
        Returns:
            Tuple of (severity_score, key_indicators_dict)
        """
        indicators = self.hazard_indicators.get(hazard_type, self.hazard_indicators["flood"])
        key_indicators = {}
        
        # Find available columns for primary indicators
        primary_cols = [col for col in indicators["primary"] if col in weather_data.columns]
        secondary_cols = [col for col in indicators["secondary"] if col in weather_data.columns]
        
        if hazard_type == "flood":
            severity = self._calculate_flood_severity(weather_data, primary_cols, secondary_cols, key_indicators)
        elif hazard_type == "drought":
            severity = self._calculate_drought_severity(weather_data, primary_cols, secondary_cols, key_indicators)
        elif hazard_type == "storm":
            severity = self._calculate_storm_severity(weather_data, primary_cols, secondary_cols, key_indicators)
        else:
            severity = 0.5  # Default unknown
        
        return float(np.clip(severity, 0, 1)), key_indicators
    
    def _calculate_flood_severity(self, data: pd.DataFrame, primary_cols: List[str], 
                                  secondary_cols: List[str], key_indicators: Dict) -> float:
        """Calculate flood severity"""
        severity = 0.0
        
        # Rainfall is primary indicator
        rainfall_col = next((c for c in primary_cols if 'rain' in c.lower() or 'precip' in c.lower()), None)
        if rainfall_col:
            max_rainfall = data[rainfall_col].max()
            accumulated = data[rainfall_col].sum()
            
            # Max rainfall factor (100mm = max)
            rainfall_factor = min(max_rainfall / 100, 1.0)
            key_indicators["max_rainfall"] = float(max_rainfall)
            
            # Accumulated rainfall matters (300mm = extreme)
            accumulated_factor = min(accumulated / 300, 1.0)
            key_indicators["accumulated_rainfall"] = float(accumulated)
            
            severity += rainfall_factor * 0.35 + accumulated_factor * 0.30
        
        # Soil saturation reduces infiltration
        soil_col = next((c for c in secondary_cols if 'soil' in c.lower()), None)
        if soil_col:
            soil_factor = data[soil_col].mean() / 100
            key_indicators["soil_moisture"] = float(data[soil_col].mean())
            severity += soil_factor * 0.20
        
        # Humidity factor
        humidity_col = next((c for c in secondary_cols if 'humid' in c.lower()), None)
        if humidity_col:
            humidity_factor = max(0, (data[humidity_col].mean() - 60)) / 40
            key_indicators["humidity"] = float(data[humidity_col].mean())
            severity += humidity_factor * 0.15
        
        return severity
    
    def _calculate_drought_severity(self, data: pd.DataFrame, primary_cols: List[str],
                                    secondary_cols: List[str], key_indicators: Dict) -> float:
        """Calculate drought severity"""
        severity = 0.0
        
        # Low rainfall over long period
        rainfall_col = next((c for c in primary_cols if 'rain' in c.lower() or 'precip' in c.lower()), None)
        if rainfall_col:
            total_rainfall = data[rainfall_col].sum()
            # Lower rainfall = higher severity
            rainfall_factor = 1 - min(total_rainfall / 50, 1.0)
            key_indicators["total_rainfall"] = float(total_rainfall)
            severity += rainfall_factor * 0.35
        
        # High temperature accelerates drought
        temp_col = next((c for c in primary_cols if 'temp' in c.lower()), None)
        if temp_col:
            avg_temp = data[temp_col].mean()
            temp_factor = min(max(0, avg_temp - 25) / 15, 1.0)
            key_indicators["avg_temperature"] = float(avg_temp)
            severity += temp_factor * 0.30
        
        # Low soil moisture
        soil_col = next((c for c in secondary_cols if 'soil' in c.lower()), None)
        if soil_col:
            soil_factor = 1 - (data[soil_col].mean() / 100)
            key_indicators["soil_moisture"] = float(data[soil_col].mean())
            severity += soil_factor * 0.20
        
        # Low humidity
        humidity_col = next((c for c in secondary_cols if 'humid' in c.lower()), None)
        if humidity_col:
            humidity_factor = 1 - min(data[humidity_col].mean() / 60, 1.0)
            key_indicators["humidity"] = float(data[humidity_col].mean())
            severity += humidity_factor * 0.15
        
        return severity
    
    def _calculate_storm_severity(self, data: pd.DataFrame, primary_cols: List[str],
                                  secondary_cols: List[str], key_indicators: Dict) -> float:
        """Calculate storm severity"""
        severity = 0.0
        
        # Wind speed is primary indicator
        wind_col = next((c for c in primary_cols if 'wind' in c.lower()), None)
        if wind_col:
            max_wind = data[wind_col].max()
            wind_factor = min(max_wind / 100, 1.0)
            key_indicators["max_wind_speed"] = float(max_wind)
            severity += wind_factor * 0.40
        
        # Low pressure indicates storm
        pressure_col = next((c for c in primary_cols if 'pressure' in c.lower()), None)
        if pressure_col:
            min_pressure = data[pressure_col].min()
            # Lower pressure = more severe (1013 is normal, 980 is severe)
            pressure_factor = max(0, (1013 - min_pressure) / 33)
            key_indicators["min_pressure"] = float(min_pressure)
            severity += min(pressure_factor, 1.0) * 0.35
        
        # Rainfall with storm
        rainfall_col = next((c for c in secondary_cols if 'rain' in c.lower() or 'precip' in c.lower()), None)
        if rainfall_col:
            max_rainfall = data[rainfall_col].max()
            rainfall_factor = min(max_rainfall / 50, 1.0)
            key_indicators["max_rainfall"] = float(max_rainfall)
            severity += rainfall_factor * 0.25
        
        return severity
    
    def _calculate_velocity(self, weather_data: pd.DataFrame, hazard_type: str) -> float:
        """
        Calculate how fast conditions are changing (0 = stable, +1 = rapidly worsening, -1 = improving)
        """
        indicators = self.hazard_indicators.get(hazard_type, self.hazard_indicators["flood"])
        primary_cols = [col for col in indicators["primary"] if col in weather_data.columns]
        
        if not primary_cols:
            return 0.0
        
        # Use primary indicator
        primary_col = primary_cols[0]
        values = weather_data[primary_col].values
        
        if len(values) < 4:
            return 0.0
        
        # Compare first half to second half
        mid = len(values) // 2
        first_half_avg = np.nanmean(values[:mid])
        second_half_avg = np.nanmean(values[mid:])
        
        # Handle edge cases
        if np.isnan(first_half_avg) or np.isnan(second_half_avg):
            return 0.0
        
        # For drought, lower rainfall = worse, so invert
        if hazard_type == "drought" and ('rain' in primary_col.lower() or 'precip' in primary_col.lower()):
            first_half_avg, second_half_avg = -first_half_avg, -second_half_avg
        
        # Normalize by first half (avoid division by zero)
        if abs(first_half_avg) > 0.1:
            velocity = (second_half_avg - first_half_avg) / (abs(first_half_avg) + 0.1)
        else:
            velocity = second_half_avg / 10  # Arbitrary scaling for low values
        
        return float(np.clip(velocity, -1, 1))
    
    def _detect_acceleration(self, weather_data: pd.DataFrame, hazard_type: str) -> float:
        """
        Is the rate of change itself accelerating?
        (Getting worse faster and faster is more dangerous)
        """
        indicators = self.hazard_indicators.get(hazard_type, self.hazard_indicators["flood"])
        primary_cols = [col for col in indicators["primary"] if col in weather_data.columns]
        
        if not primary_cols:
            return 0.0
        
        primary_col = primary_cols[0]
        values = weather_data[primary_col].values
        
        if len(values) < 4:
            return 0.0
        
        # Calculate day-to-day changes
        changes = np.diff(values)
        changes = changes[~np.isnan(changes)]
        
        if len(changes) < 2:
            return 0.0
        
        # Is the magnitude of changes increasing?
        mid = len(changes) // 2
        first_changes = np.abs(changes[:mid]).mean()
        second_changes = np.abs(changes[mid:]).mean()
        
        if first_changes < 0.1:
            acceleration = second_changes / 0.1
        else:
            acceleration = (second_changes - first_changes) / first_changes
        
        return float(np.clip(acceleration, -1, 1))
    
    def _classify_stage(self, severity: float, velocity: float) -> ProgressionStage:
        """
        Map severity score and velocity to progression stage
        
        The stage depends on both how bad things are AND how fast they're getting worse
        """
        # Adjust severity based on velocity
        # If worsening rapidly, bump up the stage
        adjusted_severity = severity + (velocity * 0.2 if velocity > 0 else 0)
        
        if adjusted_severity < self.stage_thresholds[ProgressionStage.DORMANT]:
            return ProgressionStage.DORMANT
        elif adjusted_severity < self.stage_thresholds[ProgressionStage.ESCALATING]:
            return ProgressionStage.ESCALATING
        elif adjusted_severity < self.stage_thresholds[ProgressionStage.CRITICAL]:
            return ProgressionStage.CRITICAL
        else:
            return ProgressionStage.PEAK
    
    def _estimate_days_to_peak(self,
                               severity: float,
                               velocity: float,
                               acceleration: float,
                               hazard_type: str) -> int:
        """
        Estimate how many days until peak risk
        
        If velocity is 0.2 (20% daily increase) and we're at 0.5 severity:
        - Days to reach 1.0 = log(2) / log(1.2) ≈ 3.8 days
        """
        # Accelerating = faster to peak
        adjusted_velocity = velocity * (1 + max(0, acceleration))
        
        if adjusted_velocity <= 0.01:
            return 999  # Improving or stable
        
        remaining = 1.0 - severity
        if remaining <= 0:
            return 0  # Already at peak
        
        # Linear estimate: days = remaining / velocity
        # More sophisticated: exponential growth
        try:
            if adjusted_velocity > 0:
                # Exponential estimate
                days = np.log(1.0 / max(severity, 0.1)) / np.log(1 + adjusted_velocity)
                days = max(1, int(np.ceil(abs(days))))
                
                # Cap based on hazard type
                max_days = {"flood": 7, "drought": 60, "storm": 3}.get(hazard_type, 14)
                return min(days, max_days)
        except Exception:
            pass
        
        return 7  # Default estimate
    
    def get_stage_description(self, stage: ProgressionStage) -> str:
        """Get human-readable description of a stage"""
        descriptions = {
            ProgressionStage.DORMANT: "Normal conditions - no immediate concern",
            ProgressionStage.ESCALATING: "Conditions worsening - monitor closely",
            ProgressionStage.CRITICAL: "High risk - prepare emergency response",
            ProgressionStage.PEAK: "Maximum danger - immediate action required"
        }
        return descriptions.get(stage, "Unknown stage")
    
    def get_recommendations(self, analysis: ProgressionAnalysis, hazard_type: str) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if analysis.current_stage == ProgressionStage.DORMANT:
            recommendations.append("Continue routine monitoring")
        
        elif analysis.current_stage == ProgressionStage.ESCALATING:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Review emergency plans")
            if analysis.is_accelerating:
                recommendations.append("Alert emergency services to stand by")
        
        elif analysis.current_stage == ProgressionStage.CRITICAL:
            recommendations.append("Activate emergency protocols")
            recommendations.append(f"Prepare for peak conditions in ~{analysis.days_to_peak_estimate} days")
            if hazard_type == "flood":
                recommendations.append("Check flood barriers and drainage systems")
                recommendations.append("Prepare evacuation routes")
            elif hazard_type == "storm":
                recommendations.append("Secure loose objects and structures")
                recommendations.append("Stock emergency supplies")
        
        elif analysis.current_stage == ProgressionStage.PEAK:
            recommendations.append("EMERGENCY: Implement full response")
            recommendations.append("Begin evacuation if necessary")
            recommendations.append("Deploy all emergency resources")
        
        return recommendations
