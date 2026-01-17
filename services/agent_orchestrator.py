"""
Agent Orchestrator

Manages all agent interactions and coordinates the prediction pipeline.
Provides a unified interface for making predictions using the multi-agent system.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd

from agents.pattern_detection_agent import PatternDetectionAgent
from agents.progression_analyzer_agent import ProgressionAnalyzerAgent
from agents.forecast_agent import ForecastAgent
from agents.anomaly_detection_agent import AnomalyDetectionAgent
from agents.ensemble_coordinator_agent import EnsembleCoordinatorAgent, EnsembleDecision
from services.temporal_processor import TemporalDataProcessor

logger = logging.getLogger("crisisconnect.agent_orchestrator")


class AgentOrchestrator:
    """
    Manages all agent interactions and coordinates predictions
    
    Flow:
    1. Get current weather data
    2. Pattern Detection Agent: Find historical matches
    3. Progression Analyzer Agent: Analyze current trajectory
    4. Anomaly Detection Agent: Check for unprecedented conditions
    5. Forecast Agent: Get predictions from each method
    6. Ensemble Coordinator: Combine into final decision
    """
    
    def __init__(self,
                 lstm_model=None,
                 data_dir: str = "data",
                 auto_load_models: bool = True):
        """
        Initialize orchestrator with all agents
        
        Args:
            lstm_model: Pre-trained LSTM model (optional)
            data_dir: Directory for data and model files
            auto_load_models: Whether to try loading saved models
        """
        self.data_dir = Path(data_dir)
        self.lstm_model = lstm_model
        
        # Initialize agents
        self.pattern_agent = PatternDetectionAgent()
        self.progression_agent = ProgressionAnalyzerAgent()
        self.anomaly_agent = AnomalyDetectionAgent()
        self.ensemble_agent = EnsembleCoordinatorAgent()
        
        # Forecast agent needs references to other agents
        self.forecast_agent = ForecastAgent(
            lstm_model=lstm_model,
            pattern_agent=self.pattern_agent,
            progression_agent=self.progression_agent
        )
        
        # Temporal processor for data preparation
        self.temporal_processor = TemporalDataProcessor()
        
        # Track training state
        self.is_trained = False
        self.training_stats: Dict[str, Any] = {}
        
        # Try to load saved models
        if auto_load_models:
            self._try_load_models()
        
        logger.info("AgentOrchestrator initialized")
    
    def _try_load_models(self):
        """Try to load previously saved models"""
        patterns_path = self.data_dir / "models" / "patterns.json"
        anomaly_path = self.data_dir / "models" / "anomaly_detector.pkl"
        
        try:
            if patterns_path.exists():
                self.pattern_agent.load_patterns(str(patterns_path))
                logger.info("Loaded saved patterns")
        except Exception as e:
            logger.warning(f"Could not load patterns: {e}")
        
        try:
            if anomaly_path.exists():
                self.anomaly_agent.load(str(anomaly_path))
                logger.info("Loaded saved anomaly detector")
        except Exception as e:
            logger.warning(f"Could not load anomaly detector: {e}")
    
    def train_agents(self,
                    historical_data: pd.DataFrame,
                    weather_data: pd.DataFrame = None,
                    feature_columns: List[str] = None) -> Dict[str, Any]:
        """
        Train all agents on historical data
        
        Args:
            historical_data: DataFrame with historical disaster records
            weather_data: Optional DataFrame with weather data
            feature_columns: Feature columns to use (auto-detected if None)
            
        Returns:
            Training statistics
        """
        logger.info("Training agents on historical data...")
        start_time = datetime.now()
        
        stats = {
            "started_at": start_time.isoformat(),
            "pattern_agent": {},
            "anomaly_agent": {},
            "errors": []
        }
        
        # Prepare data using temporal processor
        try:
            X, y, features = self.temporal_processor.create_sequences_from_master_dataset(
                historical_data,
                feature_columns
            )
            
            if len(X) == 0:
                stats["errors"].append("No sequences created from data")
                return stats
            
            stats["samples"] = len(X)
            stats["features"] = len(features)
            stats["feature_names"] = features
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            stats["errors"].append(f"Data preparation failed: {e}")
            return stats
        
        # Train Pattern Detection Agent
        try:
            logger.info("Training Pattern Detection Agent...")
            
            # Create DataFrame for pattern agent
            sequences_df = pd.DataFrame(X, columns=features)
            
            # Add disaster type if available
            if 'disaster_type' in historical_data.columns:
                sequences_df['disaster_type'] = historical_data['disaster_type'].values[:len(X)]
            else:
                sequences_df['disaster_type'] = 'unknown'
            
            patterns = self.pattern_agent.discover_patterns(
                sequences_df,
                y,
                features,
                disaster_type_column='disaster_type'
            )
            
            stats["pattern_agent"] = {
                "patterns_discovered": len(patterns),
                "pattern_summary": self.pattern_agent.get_pattern_summary()
            }
            
        except Exception as e:
            logger.error(f"Error training pattern agent: {e}")
            stats["errors"].append(f"Pattern agent training failed: {e}")
        
        # Train Anomaly Detection Agent
        try:
            logger.info("Training Anomaly Detection Agent...")
            
            self.anomaly_agent.train(X, features)
            
            stats["anomaly_agent"] = {
                "trained": True,
                "summary": self.anomaly_agent.get_summary()
            }
            
        except Exception as e:
            logger.error(f"Error training anomaly agent: {e}")
            stats["errors"].append(f"Anomaly agent training failed: {e}")
        
        # Save trained models
        try:
            models_dir = self.data_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            self.pattern_agent.save_patterns(str(models_dir / "patterns.json"))
            self.anomaly_agent.save(str(models_dir / "anomaly_detector.pkl"))
            
            stats["models_saved"] = True
            
        except Exception as e:
            logger.warning(f"Could not save models: {e}")
            stats["models_saved"] = False
        
        # Calculate training time
        end_time = datetime.now()
        stats["completed_at"] = end_time.isoformat()
        stats["training_time_seconds"] = (end_time - start_time).total_seconds()
        
        self.is_trained = len(stats["errors"]) == 0
        self.training_stats = stats
        
        logger.info(f"Agent training completed in {stats['training_time_seconds']:.1f}s")
        
        return stats
    
    def predict(self,
               current_weather: pd.DataFrame,
               location: str = None,
               hazard_type: str = "flood") -> Dict[str, Any]:
        """
        Full prediction pipeline with all agents
        
        Args:
            current_weather: DataFrame with recent weather data
            location: Location identifier
            hazard_type: Type of hazard to predict
            
        Returns:
            Comprehensive prediction result
        """
        start_time = datetime.now()
        
        result = {
            "location": location,
            "hazard_type": hazard_type,
            "timestamp": start_time.isoformat(),
            "prediction": None,
            "forecasts": [],
            "anomaly": None,
            "errors": []
        }
        
        try:
            # Prepare current data as sequence
            feature_cols = [c for c in current_weather.columns 
                          if c not in ['date', 'timestamp', 'location', 'country']]
            numeric_cols = current_weather[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                result["errors"].append("No numeric features in weather data")
                return result
            
            current_sequence = current_weather[numeric_cols].values
            current_sequence = np.nan_to_num(current_sequence, nan=0.0)
            
            # Stage 1: Anomaly Detection
            logger.info("  Stage 1: Checking for anomalies...")
            anomaly_result = None
            if self.anomaly_agent.is_trained:
                try:
                    # Use mean of sequence for anomaly detection
                    sequence_mean = current_sequence.mean(axis=0) if current_sequence.ndim > 1 else current_sequence
                    anomaly_result = self.anomaly_agent.detect_single(sequence_mean)
                    result["anomaly"] = {
                        "is_anomalous": anomaly_result.is_anomalous,
                        "score": anomaly_result.anomaly_score,
                        "percentile": anomaly_result.percentile,
                        "explanation": anomaly_result.explanation
                    }
                except Exception as e:
                    logger.warning(f"Anomaly detection failed: {e}")
            
            # Stage 2: Generate forecasts from all methods
            logger.info("  Stage 2: Generating forecasts...")
            forecasts = self.forecast_agent.forecast(
                current_sequence,
                current_weather,
                hazard_type,
                numeric_cols
            )
            
            result["forecasts"] = [f.to_dict() for f in forecasts]
            
            # Stage 3: Ensemble coordination
            logger.info("  Stage 3: Coordinating ensemble decision...")
            ensemble_decision = self.ensemble_agent.coordinate(
                forecasts,
                anomaly_result,
                hazard_type
            )
            
            result["prediction"] = ensemble_decision.to_dict()
            
            # Add explanation
            result["explanation"] = self.ensemble_agent.explain_decision(
                ensemble_decision, forecasts
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            result["errors"].append(str(e))
        
        # Calculate processing time
        end_time = datetime.now()
        result["processing_time_seconds"] = (end_time - start_time).total_seconds()
        
        logger.info(f"  [OK] Prediction complete ({result['processing_time_seconds']:.2f}s)")
        
        return result
    
    async def predict_async(self,
                           current_weather: pd.DataFrame,
                           location: str = None,
                           hazard_type: str = "flood") -> Dict[str, Any]:
        """
        Async version of predict for use in async contexts
        """
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.predict(current_weather, location, hazard_type)
        )
    
    def predict_batch(self,
                     locations_weather: Dict[str, pd.DataFrame],
                     hazard_type: str = "flood") -> Dict[str, Dict]:
        """
        Generate predictions for multiple locations
        
        Args:
            locations_weather: Dict mapping location_id to weather DataFrame
            hazard_type: Type of hazard to predict
            
        Returns:
            Dict mapping location_id to prediction result
        """
        results = {}
        
        for location_id, weather_data in locations_weather.items():
            try:
                result = self.predict(weather_data, location_id, hazard_type)
                results[location_id] = result
            except Exception as e:
                logger.error(f"Prediction failed for {location_id}: {e}")
                results[location_id] = {"error": str(e)}
        
        return results
    
    def set_lstm_model(self, model):
        """Update the LSTM model"""
        self.lstm_model = model
        self.forecast_agent.set_lstm_model(model)
        logger.info("LSTM model updated in orchestrator")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all agents"""
        return {
            "orchestrator": {
                "is_trained": self.is_trained,
                "training_stats": self.training_stats
            },
            "pattern_agent": {
                "is_trained": self.pattern_agent.is_trained,
                "patterns_count": len(self.pattern_agent.discovered_patterns)
            },
            "progression_agent": {
                "stage_history_length": len(self.progression_agent.stage_history)
            },
            "anomaly_agent": {
                "is_trained": self.anomaly_agent.is_trained,
                "summary": self.anomaly_agent.get_summary() if self.anomaly_agent.is_trained else None
            },
            "forecast_agent": {
                "has_lstm": self.forecast_agent.lstm_model is not None,
                "has_pattern_agent": self.forecast_agent.pattern_agent is not None,
                "has_progression_agent": self.forecast_agent.progression_agent is not None
            },
            "ensemble_agent": {
                "weights": self.ensemble_agent.weights
            }
        }
    
    def update_ensemble_weights(self, new_weights: Dict[str, float]):
        """Update ensemble agent weights based on performance"""
        self.ensemble_agent.update_weights(new_weights)
    
    def save_all_models(self, output_dir: str = None):
        """Save all trained models"""
        if output_dir is None:
            output_dir = self.data_dir / "models"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pattern agent
        if self.pattern_agent.is_trained:
            self.pattern_agent.save_patterns(str(output_dir / "patterns.json"))
        
        # Save anomaly agent
        if self.anomaly_agent.is_trained:
            self.anomaly_agent.save(str(output_dir / "anomaly_detector.pkl"))
        
        logger.info(f"All models saved to {output_dir}")
    
    def load_all_models(self, input_dir: str = None):
        """Load all trained models"""
        if input_dir is None:
            input_dir = self.data_dir / "models"
        else:
            input_dir = Path(input_dir)
        
        patterns_path = input_dir / "patterns.json"
        anomaly_path = input_dir / "anomaly_detector.pkl"
        
        if patterns_path.exists():
            self.pattern_agent.load_patterns(str(patterns_path))
            logger.info("Loaded pattern agent")
        
        if anomaly_path.exists():
            self.anomaly_agent.load(str(anomaly_path))
            logger.info("Loaded anomaly agent")
        
        self.is_trained = (
            self.pattern_agent.is_trained or 
            self.anomaly_agent.is_trained
        )


def create_orchestrator(lstm_model=None, data_dir: str = "data") -> AgentOrchestrator:
    """
    Factory function to create AgentOrchestrator
    
    Args:
        lstm_model: Pre-trained LSTM model
        data_dir: Data directory path
        
    Returns:
        Configured AgentOrchestrator instance
    """
    return AgentOrchestrator(
        lstm_model=lstm_model,
        data_dir=data_dir,
        auto_load_models=True
    )
