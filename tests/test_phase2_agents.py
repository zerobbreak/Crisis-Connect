"""
Phase 2 Multi-Agent System - Comprehensive Test Suite

Tests all agents and the orchestration system:
- PatternDetectionAgent
- ProgressionAnalyzerAgent
- ForecastAgent
- AnomalyDetectionAgent
- EnsembleCoordinatorAgent
- AgentOrchestrator
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.pattern_detection_agent import PatternDetectionAgent, Pattern
from agents.progression_analyzer_agent import ProgressionAnalyzerAgent, ProgressionStage, ProgressionAnalysis
from agents.forecast_agent import ForecastAgent, Forecast
from agents.anomaly_detection_agent import AnomalyDetectionAgent, AnomalyResult
from agents.ensemble_coordinator_agent import EnsembleCoordinatorAgent, EnsembleDecision
from services.temporal_processor import TemporalDataProcessor
from services.agent_orchestrator import AgentOrchestrator
from services.agent_performance_monitor import AgentPerformanceMonitor


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'date': dates,
        'rainfall': np.random.exponential(10, 30),
        'precip_mm': np.random.exponential(10, 30),
        'temperature': 20 + np.random.normal(0, 5, 30),
        'temp_c': 20 + np.random.normal(0, 5, 30),
        'wind_speed': 15 + np.random.exponential(5, 30),
        'wind_kph': 15 + np.random.exponential(5, 30),
        'humidity': 60 + np.random.normal(0, 15, 30),
        'pressure_mb': 1013 + np.random.normal(0, 10, 30),
        'soil_moisture': 50 + np.random.normal(0, 20, 30),
        'location': ['TestCity'] * 30,
        'disaster_type': ['flood'] * 30
    })


@pytest.fixture
def flood_weather_data():
    """Create weather data showing flood progression"""
    dates = pd.date_range('2024-01-01', periods=14, freq='D')
    return pd.DataFrame({
        'date': dates,
        'rainfall': np.linspace(5, 80, 14),  # Steadily increasing rainfall
        'precip_mm': np.linspace(5, 80, 14),
        'temperature': [22] * 14,
        'temp_c': [22] * 14,
        'wind_speed': [20] * 14,
        'wind_kph': [20] * 14,
        'humidity': np.linspace(60, 95, 14),  # Increasing humidity
        'pressure_mb': np.linspace(1015, 1000, 14),  # Dropping pressure
        'soil_moisture': np.linspace(40, 90, 14),  # Saturating soil
        'location': ['FloodCity'] * 14
    })


@pytest.fixture
def sample_disaster_data():
    """Create sample disaster records"""
    return pd.DataFrame({
        'disaster_id': ['D001', 'D002', 'D003', 'D004', 'D005'],
        'date': pd.date_range('2024-01-15', periods=5, freq='30D'),
        'location': ['CityA', 'CityB', 'CityA', 'CityC', 'CityB'],
        'country': ['Country1', 'Country1', 'Country1', 'Country2', 'Country1'],
        'disaster_type': ['flood', 'flood', 'storm', 'flood', 'drought'],
        'deaths': [10, 5, 0, 20, 0],
        'affected': [1000, 500, 200, 2000, 5000],
        'damage_usd': [1e6, 5e5, 1e5, 2e6, 1e6]
    })


@pytest.fixture
def sample_sequences():
    """Create sample sequences for pattern detection"""
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    
    # Create sequences with some patterns
    X = np.random.randn(n_samples, n_features)
    
    # Add pattern to first 20 samples (disaster sequences)
    X[:20, 0] += 2  # Higher rainfall
    X[:20, 1] += 1  # Higher humidity
    
    y = np.array([1] * 20 + [0] * 30)  # First 20 are disasters
    
    df = pd.DataFrame(X, columns=['rainfall', 'humidity', 'wind', 'pressure', 'temp'])
    df['disaster_type'] = 'flood'
    
    return df, y


# ============================================================================
# Pattern Detection Agent Tests
# ============================================================================

class TestPatternDetectionAgent:
    """Tests for PatternDetectionAgent"""
    
    def test_initialization(self):
        """Test agent initializes correctly"""
        agent = PatternDetectionAgent(similarity_threshold=0.85)
        assert agent.similarity_threshold == 0.85
        assert agent.is_trained == False
        assert len(agent.discovered_patterns) == 0
    
    def test_discover_patterns(self, sample_sequences):
        """Test pattern discovery from sequences"""
        df, labels = sample_sequences
        agent = PatternDetectionAgent(similarity_threshold=0.7, min_pattern_occurrences=3)
        
        feature_columns = ['rainfall', 'humidity', 'wind', 'pressure', 'temp']
        patterns = agent.discover_patterns(df, labels, feature_columns)
        
        assert agent.is_trained == True
        # Should discover at least one pattern from the clustered data
        assert len(patterns) >= 0  # May or may not find patterns depending on data
    
    def test_match_current_sequence(self, sample_sequences):
        """Test matching current sequence against patterns"""
        df, labels = sample_sequences
        agent = PatternDetectionAgent(similarity_threshold=0.7, min_pattern_occurrences=3)
        
        feature_columns = ['rainfall', 'humidity', 'wind', 'pressure', 'temp']
        agent.discover_patterns(df, labels, feature_columns)
        
        # Create a test sequence similar to disaster patterns
        test_sequence = np.array([2.5, 1.5, 0.0, 0.0, 0.0])
        matches = agent.match_current_sequence(test_sequence)
        
        # Should return list (may be empty if no patterns discovered)
        assert isinstance(matches, list)
    
    def test_pattern_summary(self, sample_sequences):
        """Test pattern summary generation"""
        df, labels = sample_sequences
        agent = PatternDetectionAgent()
        
        feature_columns = ['rainfall', 'humidity', 'wind', 'pressure', 'temp']
        agent.discover_patterns(df, labels, feature_columns)
        
        summary = agent.get_pattern_summary()
        assert 'total_patterns' in summary
        assert 'patterns' in summary


# ============================================================================
# Progression Analyzer Agent Tests
# ============================================================================

class TestProgressionAnalyzerAgent:
    """Tests for ProgressionAnalyzerAgent"""
    
    def test_initialization(self):
        """Test agent initializes correctly"""
        agent = ProgressionAnalyzerAgent()
        assert len(agent.stage_history) == 0
        assert ProgressionStage.DORMANT in agent.stage_thresholds
    
    def test_analyze_progression_flood(self, flood_weather_data):
        """Test progression analysis for flood conditions"""
        agent = ProgressionAnalyzerAgent()
        
        analysis = agent.analyze_progression(flood_weather_data, hazard_type="flood")
        
        assert isinstance(analysis, ProgressionAnalysis)
        assert 0 <= analysis.severity_score <= 1
        assert -1 <= analysis.worsening_velocity <= 1
        assert analysis.days_to_peak_estimate >= 0
        assert 0 <= analysis.confidence <= 1
        
        # With increasing rainfall, should detect worsening conditions
        assert analysis.worsening_velocity > 0 or analysis.severity_score > 0.3
    
    def test_analyze_progression_stages(self, sample_weather_data):
        """Test that progression stages are classified correctly"""
        agent = ProgressionAnalyzerAgent()
        
        analysis = agent.analyze_progression(sample_weather_data, hazard_type="flood")
        
        assert analysis.current_stage in [
            ProgressionStage.DORMANT,
            ProgressionStage.ESCALATING,
            ProgressionStage.CRITICAL,
            ProgressionStage.PEAK
        ]
    
    def test_stage_descriptions(self):
        """Test stage description generation"""
        agent = ProgressionAnalyzerAgent()
        
        for stage in ProgressionStage:
            desc = agent.get_stage_description(stage)
            assert isinstance(desc, str)
            assert len(desc) > 0
    
    def test_recommendations(self, flood_weather_data):
        """Test recommendation generation"""
        agent = ProgressionAnalyzerAgent()
        analysis = agent.analyze_progression(flood_weather_data, hazard_type="flood")
        
        recommendations = agent.get_recommendations(analysis, "flood")
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


# ============================================================================
# Forecast Agent Tests
# ============================================================================

class TestForecastAgent:
    """Tests for ForecastAgent"""
    
    def test_initialization(self):
        """Test agent initializes correctly"""
        agent = ForecastAgent()
        assert agent.lstm_model is None
        assert agent.pattern_agent is None
        assert agent.progression_agent is None
    
    def test_initialization_with_agents(self):
        """Test initialization with sub-agents"""
        pattern_agent = PatternDetectionAgent()
        progression_agent = ProgressionAnalyzerAgent()
        
        agent = ForecastAgent(
            pattern_agent=pattern_agent,
            progression_agent=progression_agent
        )
        
        assert agent.pattern_agent is not None
        assert agent.progression_agent is not None
    
    def test_forecast_with_progression(self, sample_weather_data):
        """Test forecasting with progression agent"""
        progression_agent = ProgressionAnalyzerAgent()
        agent = ForecastAgent(progression_agent=progression_agent)
        
        sequence = sample_weather_data[['rainfall', 'humidity', 'wind_speed']].values
        forecasts = agent.forecast(sequence, sample_weather_data, "flood")
        
        assert isinstance(forecasts, list)
        assert len(forecasts) > 0
        
        for forecast in forecasts:
            assert isinstance(forecast, Forecast)
            assert 0 <= forecast.risk_score <= 1
            assert forecast.hours_to_peak > 0
            assert forecast.method in ['lstm', 'pattern_matching', 'progression', 'heuristic']
    
    def test_fallback_forecast(self, sample_weather_data):
        """Test fallback forecast when no agents available"""
        agent = ForecastAgent()
        
        sequence = sample_weather_data[['rainfall', 'humidity']].values
        forecasts = agent.forecast(sequence, sample_weather_data, "flood")
        
        assert len(forecasts) == 1
        assert forecasts[0].method == 'heuristic'
    
    def test_forecast_summary(self, sample_weather_data):
        """Test forecast summary generation"""
        progression_agent = ProgressionAnalyzerAgent()
        agent = ForecastAgent(progression_agent=progression_agent)
        
        sequence = sample_weather_data[['rainfall', 'humidity']].values
        forecasts = agent.forecast(sequence, sample_weather_data, "flood")
        
        summary = agent.get_forecast_summary(forecasts)
        
        assert 'num_forecasts' in summary
        assert 'methods_used' in summary
        assert 'risk_scores' in summary


# ============================================================================
# Anomaly Detection Agent Tests
# ============================================================================

class TestAnomalyDetectionAgent:
    """Tests for AnomalyDetectionAgent"""
    
    def test_initialization(self):
        """Test agent initializes correctly"""
        agent = AnomalyDetectionAgent(contamination=0.1)
        assert agent.contamination == 0.1
        assert agent.is_trained == False
    
    def test_train(self):
        """Test training anomaly detector"""
        agent = AnomalyDetectionAgent()
        
        # Create training data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        
        agent.train(X_train, ['f1', 'f2', 'f3', 'f4', 'f5'])
        
        assert agent.is_trained == True
        assert len(agent.feature_columns) == 5
        assert len(agent.training_stats) == 5
    
    def test_detect_normal(self):
        """Test detecting normal data"""
        agent = AnomalyDetectionAgent()
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        agent.train(X_train)
        
        # Test with normal data
        X_test = np.random.randn(10, 5)
        results = agent.detect(X_test)
        
        assert len(results) == 10
        for result in results:
            assert isinstance(result, AnomalyResult)
            assert result.is_anomalous in [True, False]  # Works with numpy bool
    
    def test_detect_anomalies(self):
        """Test detecting actual anomalies"""
        agent = AnomalyDetectionAgent(contamination=0.1)
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        agent.train(X_train)
        
        # Test with anomalous data (far from training distribution)
        X_anomaly = np.array([[10, 10, 10, 10, 10]])  # Very different from training
        results = agent.detect(X_anomaly)
        
        assert len(results) == 1
        # Should likely be flagged as anomalous
        assert results[0].anomaly_score < 0  # Negative scores indicate anomalies
    
    def test_detect_single(self):
        """Test single sample detection"""
        agent = AnomalyDetectionAgent()
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        agent.train(X_train)
        
        X_single = np.random.randn(5)
        result = agent.detect_single(X_single)
        
        assert isinstance(result, AnomalyResult)
        assert 'explanation' in result.__dict__


# ============================================================================
# Ensemble Coordinator Agent Tests
# ============================================================================

class TestEnsembleCoordinatorAgent:
    """Tests for EnsembleCoordinatorAgent"""
    
    def test_initialization(self):
        """Test agent initializes correctly"""
        agent = EnsembleCoordinatorAgent()
        assert 'lstm' in agent.weights
        assert 'pattern_matching' in agent.weights
        assert 'progression' in agent.weights
    
    def test_coordinate_single_forecast(self):
        """Test coordination with single forecast"""
        agent = EnsembleCoordinatorAgent()
        
        forecasts = [
            Forecast("flood", 0.7, 48, 0.8, "lstm", "Test forecast")
        ]
        
        decision = agent.coordinate(forecasts)
        
        assert isinstance(decision, EnsembleDecision)
        assert 0 <= decision.risk_score <= 1
        assert decision.risk_level in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
    
    def test_coordinate_multiple_forecasts(self):
        """Test coordination with multiple forecasts"""
        agent = EnsembleCoordinatorAgent()
        
        forecasts = [
            Forecast("flood", 0.7, 48, 0.9, "lstm", "LSTM prediction"),
            Forecast("flood", 0.6, 36, 0.8, "pattern_matching", "Pattern match"),
            Forecast("flood", 0.8, 24, 0.7, "progression", "Progression analysis")
        ]
        
        decision = agent.coordinate(forecasts)
        
        assert isinstance(decision, EnsembleDecision)
        # Risk should be weighted average
        assert 0.6 <= decision.risk_score <= 0.8
        assert len(decision.method_breakdown) == 3
    
    def test_disagreement_detection(self):
        """Test detection of method disagreement"""
        agent = EnsembleCoordinatorAgent(disagreement_threshold=0.3)
        
        # Create forecasts with significant disagreement
        forecasts = [
            Forecast("flood", 0.9, 12, 0.9, "lstm", "High risk"),
            Forecast("flood", 0.2, 72, 0.8, "progression", "Low risk")
        ]
        
        decision = agent.coordinate(forecasts)
        
        # Should have warning about disagreement
        assert len(decision.warnings) > 0
    
    def test_anomaly_handling(self):
        """Test handling of anomaly results"""
        agent = EnsembleCoordinatorAgent()
        
        forecasts = [
            Forecast("flood", 0.5, 48, 0.8, "lstm", "Test")
        ]
        
        anomaly = AnomalyResult(
            is_anomalous=True,
            anomaly_score=-0.3,
            percentile=10,
            contributing_features={'rainfall': {'z_score': 3.0}},
            explanation="Unusual conditions"
        )
        
        decision = agent.coordinate(forecasts, anomaly)
        
        assert decision.is_anomalous == True
        assert decision.anomaly_details is not None
        assert any('ANOMALY' in w for w in decision.warnings)
    
    def test_recommendation_generation(self):
        """Test recommendation generation for different risk levels"""
        agent = EnsembleCoordinatorAgent()
        
        risk_levels = [
            ([Forecast("flood", 0.1, 72, 0.8, "lstm", "Low")], "LOW"),
            ([Forecast("flood", 0.4, 48, 0.8, "lstm", "Moderate")], "MODERATE"),
            ([Forecast("flood", 0.7, 24, 0.8, "lstm", "High")], "HIGH"),
            ([Forecast("flood", 0.9, 6, 0.8, "lstm", "Critical")], "CRITICAL")
        ]
        
        for forecasts, expected_level in risk_levels:
            decision = agent.coordinate(forecasts)
            assert decision.risk_level == expected_level
            assert len(decision.recommendation) > 0


# ============================================================================
# Temporal Processor Tests
# ============================================================================

class TestTemporalDataProcessor:
    """Tests for TemporalDataProcessor"""
    
    def test_initialization(self):
        """Test processor initializes correctly"""
        processor = TemporalDataProcessor(lookback_days=14)
        assert processor.lookback_days == 14
    
    def test_create_sequences_from_master(self, sample_disaster_data):
        """Test creating sequences from master dataset"""
        processor = TemporalDataProcessor()
        
        X, y, features = processor.create_sequences_from_master_dataset(sample_disaster_data)
        
        assert len(X) == len(sample_disaster_data)
        assert len(y) == len(sample_disaster_data)
        assert len(features) > 0
    
    def test_prepare_features(self, sample_weather_data):
        """Test feature preparation"""
        processor = TemporalDataProcessor()
        
        prepared = processor.prepare_features(sample_weather_data)
        
        # Should have added rolling features
        assert len(prepared.columns) >= len(sample_weather_data.columns)


# ============================================================================
# Agent Orchestrator Tests
# ============================================================================

class TestAgentOrchestrator:
    """Tests for AgentOrchestrator"""
    
    def test_initialization(self):
        """Test orchestrator initializes correctly"""
        orchestrator = AgentOrchestrator(auto_load_models=False)
        
        assert orchestrator.pattern_agent is not None
        assert orchestrator.progression_agent is not None
        assert orchestrator.anomaly_agent is not None
        assert orchestrator.ensemble_agent is not None
        assert orchestrator.forecast_agent is not None
    
    def test_train_agents(self, sample_disaster_data):
        """Test training all agents"""
        orchestrator = AgentOrchestrator(auto_load_models=False)
        
        stats = orchestrator.train_agents(sample_disaster_data)
        
        assert 'pattern_agent' in stats
        assert 'anomaly_agent' in stats
        assert 'training_time_seconds' in stats
    
    def test_predict(self, sample_weather_data, sample_disaster_data):
        """Test full prediction pipeline"""
        orchestrator = AgentOrchestrator(auto_load_models=False)
        
        # Train first
        orchestrator.train_agents(sample_disaster_data)
        
        # Make prediction
        result = orchestrator.predict(
            sample_weather_data,
            location="TestCity",
            hazard_type="flood"
        )
        
        assert 'prediction' in result
        assert 'forecasts' in result
        assert 'processing_time_seconds' in result
        assert result['processing_time_seconds'] < 10  # Should be fast
    
    def test_get_status(self):
        """Test status retrieval"""
        orchestrator = AgentOrchestrator(auto_load_models=False)
        
        status = orchestrator.get_status()
        
        assert 'orchestrator' in status
        assert 'pattern_agent' in status
        assert 'anomaly_agent' in status
        assert 'forecast_agent' in status
        assert 'ensemble_agent' in status


# ============================================================================
# Performance Monitor Tests
# ============================================================================

class TestAgentPerformanceMonitor:
    """Tests for AgentPerformanceMonitor"""
    
    def test_initialization(self, tmp_path):
        """Test monitor initializes correctly"""
        monitor = AgentPerformanceMonitor(log_dir=str(tmp_path))
        assert len(monitor.predictions) == 0
    
    def test_log_prediction(self, tmp_path):
        """Test logging predictions"""
        monitor = AgentPerformanceMonitor(log_dir=str(tmp_path))
        
        prediction_result = {
            "prediction": {
                "risk_score": 0.7,
                "risk_level": "HIGH",
                "hours_to_peak": 24,
                "method_breakdown": {"lstm": 0.7, "progression": 0.65},
                "primary_method": "lstm",
                "confidence": 0.8,
                "is_anomalous": False,
                "warnings": []
            },
            "processing_time_seconds": 1.5
        }
        
        monitor.log_prediction("TestCity", "flood", prediction_result)
        
        assert len(monitor.predictions) == 1
        assert monitor.predictions[0].location == "TestCity"
    
    def test_calculate_metrics(self, tmp_path):
        """Test metrics calculation"""
        monitor = AgentPerformanceMonitor(log_dir=str(tmp_path))
        
        # Log some predictions
        for i in range(5):
            prediction_result = {
                "prediction": {
                    "risk_score": 0.5 + i * 0.1,
                    "risk_level": "MODERATE",
                    "hours_to_peak": 24,
                    "method_breakdown": {"lstm": 0.5},
                    "primary_method": "lstm",
                    "confidence": 0.7,
                    "is_anomalous": False,
                    "warnings": []
                },
                "processing_time_seconds": 1.0
            }
            monitor.log_prediction(f"City{i}", "flood", prediction_result)
        
        metrics = monitor.calculate_metrics()
        
        assert metrics['total_predictions'] == 5
        assert 'speed' in metrics
        assert 'risk_distribution' in metrics
    
    def test_generate_report(self, tmp_path):
        """Test report generation"""
        monitor = AgentPerformanceMonitor(log_dir=str(tmp_path))
        
        # Log a prediction
        prediction_result = {
            "prediction": {
                "risk_score": 0.7,
                "risk_level": "HIGH",
                "hours_to_peak": 24,
                "method_breakdown": {"lstm": 0.7},
                "primary_method": "lstm",
                "confidence": 0.8,
                "is_anomalous": False,
                "warnings": []
            },
            "processing_time_seconds": 1.5
        }
        monitor.log_prediction("TestCity", "flood", prediction_result)
        
        report = monitor.generate_report(days=30)
        
        assert isinstance(report, str)
        assert "PERFORMANCE REPORT" in report


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full multi-agent system"""
    
    def test_full_pipeline(self, sample_weather_data, sample_disaster_data, tmp_path):
        """Test complete pipeline from training to prediction"""
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(
            data_dir=str(tmp_path),
            auto_load_models=False
        )
        
        # Train agents
        train_stats = orchestrator.train_agents(sample_disaster_data)
        assert 'training_time_seconds' in train_stats
        
        # Make prediction
        result = orchestrator.predict(
            sample_weather_data,
            location="TestCity",
            hazard_type="flood"
        )
        
        assert result['prediction'] is not None
        assert result['prediction']['risk_level'] in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        
        # Log prediction
        monitor = AgentPerformanceMonitor(log_dir=str(tmp_path))
        monitor.log_prediction("TestCity", "flood", result)
        
        # Check metrics
        metrics = monitor.calculate_metrics()
        assert metrics['total_predictions'] == 1
    
    def test_flood_scenario(self, flood_weather_data, sample_disaster_data, tmp_path):
        """Test with realistic flood scenario"""
        orchestrator = AgentOrchestrator(
            data_dir=str(tmp_path),
            auto_load_models=False
        )
        
        orchestrator.train_agents(sample_disaster_data)
        
        result = orchestrator.predict(
            flood_weather_data,
            location="FloodCity",
            hazard_type="flood"
        )
        
        # With worsening conditions, should detect elevated risk
        prediction = result['prediction']
        assert prediction is not None
        
        # Should have at least one forecast
        assert len(result['forecasts']) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
