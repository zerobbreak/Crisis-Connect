"""
Anomaly Detection Agent

Detects unusual weather patterns not seen before using Isolation Forest.
Critical for catching unprecedented conditions that traditional models miss.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Anomaly detection will be limited.")

logger = logging.getLogger("crisisconnect.agents.anomaly_detection")


@dataclass
class AnomalyResult:
    """Result of anomaly detection"""
    is_anomalous: bool
    anomaly_score: float  # -1 to 1, more negative = more anomalous
    percentile: float  # How unusual compared to training data (0-100)
    contributing_features: Dict[str, float]  # Which features are most unusual
    explanation: str


class AnomalyDetectionAgent:
    """
    Agent that detects unusual patterns not seen before
    
    Why this matters:
    - Climate change creates unprecedented weather patterns
    - Traditional models fail on "never before seen" conditions
    - Anomaly detection catches these early
    - Provides early warning even when patterns don't match historical disasters
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """
        Initialize anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies in training data (0.01-0.5)
            n_estimators: Number of trees in the forest
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for AnomalyDetectionAgent")
        
        self.contamination = contamination
        self.n_estimators = n_estimators
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns: List[str] = []
        self.training_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"AnomalyDetectionAgent initialized: contamination={contamination}, n_estimators={n_estimators}")
    
    def train(self, X: np.ndarray, feature_columns: List[str] = None):
        """
        Train anomaly detector on normal/historical data
        
        Args:
            X: Training data (n_samples, n_features)
            feature_columns: Optional list of feature names
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store feature columns
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate and store training statistics for each feature
        for i, col in enumerate(self.feature_columns):
            self.training_stats[col] = {
                "mean": float(np.mean(X[:, i])),
                "std": float(np.std(X[:, i])),
                "min": float(np.min(X[:, i])),
                "max": float(np.max(X[:, i])),
                "q25": float(np.percentile(X[:, i], 25)),
                "q75": float(np.percentile(X[:, i], 75))
            }
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        self.model.fit(X_scaled)
        self.is_trained = True
        
        logger.info(f"Anomaly detector trained on {len(X)} samples with {X.shape[1]} features")
    
    def detect(self, X: np.ndarray) -> List[AnomalyResult]:
        """
        Detect anomalies in new data
        
        Args:
            X: Data to check (n_samples, n_features)
            
        Returns:
            List of AnomalyResult for each sample
        """
        if not self.is_trained:
            logger.warning("Anomaly detector not trained, returning default results")
            return [AnomalyResult(
                is_anomalous=False,
                anomaly_score=0.0,
                percentile=50.0,
                contributing_features={},
                explanation="Anomaly detector not trained"
            ) for _ in range(len(X) if X.ndim > 1 else 1)]
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Get predictions (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_scaled)
        
        # Get anomaly scores (more negative = more anomalous)
        scores = self.model.score_samples(X_scaled)
        
        results = []
        for i in range(len(X)):
            is_anomalous = predictions[i] == -1
            anomaly_score = float(scores[i])
            
            # Convert score to percentile (approximate)
            # Scores typically range from -0.5 (very anomalous) to 0.1 (very normal)
            percentile = self._score_to_percentile(anomaly_score)
            
            # Find contributing features
            contributing = self._find_contributing_features(X[i])
            
            # Generate explanation
            explanation = self._generate_explanation(is_anomalous, percentile, contributing)
            
            results.append(AnomalyResult(
                is_anomalous=is_anomalous,
                anomaly_score=anomaly_score,
                percentile=percentile,
                contributing_features=contributing,
                explanation=explanation
            ))
        
        return results
    
    def detect_single(self, X: np.ndarray) -> AnomalyResult:
        """
        Detect anomaly in a single sample
        
        Args:
            X: Single sample (1D or 2D array)
            
        Returns:
            AnomalyResult for the sample
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        results = self.detect(X)
        return results[0]
    
    def _score_to_percentile(self, score: float) -> float:
        """
        Convert isolation forest score to percentile
        
        Scores typically range from about -0.5 (anomalous) to 0.1 (normal)
        """
        # Normalize to 0-100 range
        # Lower scores = more anomalous = lower percentile
        normalized = (score + 0.5) / 0.6  # Rough normalization
        percentile = max(0, min(100, normalized * 100))
        return percentile
    
    def _find_contributing_features(self, sample: np.ndarray) -> Dict[str, float]:
        """
        Find which features contribute most to the anomaly
        
        Uses z-score to identify features that deviate most from training distribution
        """
        contributing = {}
        
        for i, col in enumerate(self.feature_columns):
            if col not in self.training_stats:
                continue
            
            stats = self.training_stats[col]
            value = sample[i]
            
            # Calculate z-score
            if stats["std"] > 0:
                z_score = abs(value - stats["mean"]) / stats["std"]
            else:
                z_score = 0.0
            
            # Only include features with significant deviation
            if z_score > 1.5:  # More than 1.5 standard deviations
                contributing[col] = {
                    "value": float(value),
                    "z_score": float(z_score),
                    "expected_range": f"{stats['q25']:.2f} - {stats['q75']:.2f}",
                    "deviation": "high" if value > stats["mean"] else "low"
                }
        
        # Sort by z-score and return top 5
        sorted_features = sorted(contributing.items(), key=lambda x: x[1]["z_score"], reverse=True)
        return dict(sorted_features[:5])
    
    def _generate_explanation(self, is_anomalous: bool, percentile: float, 
                             contributing: Dict[str, float]) -> str:
        """Generate human-readable explanation"""
        if not is_anomalous:
            return f"Conditions are within normal range (percentile: {percentile:.0f}%)"
        
        explanation = f"ANOMALY DETECTED: Conditions are unusual (percentile: {percentile:.0f}%)"
        
        if contributing:
            top_features = list(contributing.keys())[:3]
            explanation += f". Key unusual factors: {', '.join(top_features)}"
        
        return explanation
    
    def get_anomaly_threshold(self) -> float:
        """Get the current anomaly score threshold"""
        if not self.is_trained:
            return 0.0
        
        # The threshold is implicitly defined by contamination
        # Approximate the threshold score
        return -0.5 + (0.5 * self.contamination)
    
    def save(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            logger.warning("Cannot save untrained model")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "training_stats": self.training_stats,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Anomaly detector saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data["model"]
        self.scaler = save_data["scaler"]
        self.feature_columns = save_data["feature_columns"]
        self.training_stats = save_data["training_stats"]
        self.contamination = save_data.get("contamination", 0.1)
        self.n_estimators = save_data.get("n_estimators", 100)
        self.is_trained = True
        
        logger.info(f"Anomaly detector loaded from {filepath}")
    
    def get_summary(self) -> Dict:
        """Get summary of the anomaly detector"""
        return {
            "is_trained": self.is_trained,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "training_stats_available": len(self.training_stats) > 0
        }
