"""
Crisis Connect Multi-Agent System

This module contains specialized agents for disaster pattern detection,
progression analysis, forecasting, anomaly detection, and ensemble coordination.

Agents:
- PatternDetectionAgent: Discovers recurring weather patterns preceding disasters
- ProgressionAnalyzerAgent: Tracks disaster development stages
- ForecastAgent: Generates predictions from multiple methods
- AnomalyDetectionAgent: Detects unprecedented weather patterns
- EnsembleCoordinatorAgent: Combines predictions from all agents
"""

from agents.pattern_detection_agent import PatternDetectionAgent, Pattern
from agents.progression_analyzer_agent import ProgressionAnalyzerAgent, ProgressionStage, ProgressionAnalysis
from agents.forecast_agent import ForecastAgent, Forecast
from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.ensemble_coordinator_agent import EnsembleCoordinatorAgent, EnsembleDecision

__all__ = [
    "PatternDetectionAgent",
    "Pattern",
    "ProgressionAnalyzerAgent",
    "ProgressionStage",
    "ProgressionAnalysis",
    "ForecastAgent",
    "Forecast",
    "AnomalyDetectionAgent",
    "EnsembleCoordinatorAgent",
    "EnsembleDecision",
]
