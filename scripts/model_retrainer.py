#!/usr/bin/env python3
"""
Model Retrainer - Retrain ML model with new data
Trains new model versions, validates performance, and manages deployment
"""
import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import structlog

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class ModelRetrainer:
    """Retrain and validate ML models"""
    
    def __init__(self, 
                 current_model_path: str = None,
                 models_dir: str = "models",
                 min_improvement: float = 0.02):
        self.current_model_path = current_model_path or settings.model_path
        self.models_dir = models_dir
        self.min_improvement = min_improvement
        self.current_model = None
        self.new_model = None
        
        # Create directories
        os.makedirs(f"{models_dir}/versions", exist_ok=True)
        os.makedirs(f"{models_dir}/backups", exist_ok=True)
        
    def load_current_model(self) -> bool:
        """Load the current production model"""
        try:
            if os.path.exists(self.current_model_path):
                self.current_model = joblib.load(self.current_model_path)
                logger.info("[OK] Loaded current model", path=self.current_model_path)
                return True
            else:
                logger.warning("[WARN] No current model found", path=self.current_model_path)
                return False
        except Exception as e:
            logger.error("[ERROR] Failed to load current model", error=str(e))
            return False
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X features, y labels)
        """
        # Feature columns (adjust based on your actual features)
        feature_cols = [
            'temperature_2m', 'relative_humidity_2m', 'precipitation',
            'wind_speed_10m', 'wind_gusts_10m', 'pressure_msl',
            'cloud_cover', 'visibility', 'lat', 'lon'
        ]
        
        # Check which features exist
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            logger.error("[ERROR] No feature columns found in dataset")
            return None, None
        
        logger.info(f"[FEATURES] Using {len(available_features)} features", features=available_features)
        
        X = df[available_features].copy()
        
        # Create labels if composite_risk_score exists
        if 'composite_risk_score' in df.columns:
            # Convert risk scores to binary classification (high risk vs low risk)
            y = (df['composite_risk_score'] >= 70).astype(int)
            logger.info(f"[LABELS] Created labels from risk scores", 
                       high_risk_count=y.sum(), 
                       low_risk_count=(~y.astype(bool)).sum())
        else:
            logger.warning("[WARN] No risk scores found, cannot create labels")
            return X, None
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """
        Train a new model
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Trained model
        """
        logger.info("[TRAIN] Starting model training", samples=len(X))
        
        # Create model with same parameters as original
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(X, y)
        
        logger.info("[OK] Model training complete")
        
        return model
    
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Validate model performance
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info("[VALIDATE] Evaluating model performance")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        metrics['cv_accuracy_mean'] = cv_scores.mean()
        metrics['cv_accuracy_std'] = cv_scores.std()
        
        logger.info("[OK] Validation complete", **metrics)
        
        return metrics
    
    def compare_models(self, new_metrics: Dict[str, float], current_metrics: Dict[str, float] = None) -> bool:
        """
        Compare new model with current model
        
        Args:
            new_metrics: Metrics for new model
            current_metrics: Metrics for current model
            
        Returns:
            True if new model is better
        """
        if not current_metrics:
            logger.info("[COMPARE] No current model to compare, accepting new model")
            return True
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"{'Metric':<20} {'Current':<15} {'New':<15} {'Change':<15}")
        print("-" * 80)
        
        improvements = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            current_val = current_metrics.get(metric, 0)
            new_val = new_metrics.get(metric, 0)
            change = new_val - current_val
            improvements[metric] = change
            
            change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
            print(f"{metric:<20} {current_val:<15.4f} {new_val:<15.4f} {change_str:<15}")
        
        print("=" * 80)
        
        # Decision logic
        accuracy_improvement = improvements['accuracy']
        precision_degradation = improvements['precision']
        recall_degradation = improvements['recall']
        
        # Must improve accuracy by at least min_improvement
        if accuracy_improvement < self.min_improvement:
            logger.warning(f"[COMPARE] Insufficient improvement: {accuracy_improvement:.4f} < {self.min_improvement}")
            print(f"\n[DECISION] REJECT - Accuracy improvement ({accuracy_improvement:.4f}) below threshold ({self.min_improvement})")
            return False
        
        # Must not significantly degrade precision or recall
        if precision_degradation < -0.05 or recall_degradation < -0.05:
            logger.warning("[COMPARE] Significant degradation in precision or recall")
            print(f"\n[DECISION] REJECT - Significant degradation in precision or recall")
            return False
        
        logger.info("[COMPARE] New model is better", improvement=accuracy_improvement)
        print(f"\n[DECISION] ACCEPT - New model shows {accuracy_improvement:.4f} improvement")
        return True
    
    def save_model(self, model, version: str = None, metrics: Dict[str, float] = None):
        """
        Save model with version
        
        Args:
            model: Model to save
            version: Version string
            metrics: Performance metrics
        """
        if not version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save versioned model
        version_path = f"{self.models_dir}/versions/model_v{version}.pkl"
        joblib.dump(model, version_path)
        logger.info("[SAVE] Saved versioned model", path=version_path)
        
        # Save metadata
        metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "model_path": version_path
        }
        
        metadata_path = f"{self.models_dir}/versions/model_v{version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("[SAVE] Saved model metadata", path=metadata_path)
        
        return version_path
    
    def deploy_model(self, model_path: str):
        """
        Deploy new model to production
        
        Args:
            model_path: Path to new model
        """
        # Backup current model
        if os.path.exists(self.current_model_path):
            backup_path = f"{self.models_dir}/backups/model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            import shutil
            shutil.copy2(self.current_model_path, backup_path)
            logger.info("[BACKUP] Backed up current model", path=backup_path)
        
        # Deploy new model
        import shutil
        os.makedirs(os.path.dirname(self.current_model_path), exist_ok=True)
        shutil.copy2(model_path, self.current_model_path)
        logger.info("[DEPLOY] Deployed new model", path=self.current_model_path)
        
        print(f"\n[SUCCESS] Model deployed to: {self.current_model_path}")
    
    def retrain(self, data_path: str, auto_deploy: bool = False) -> Dict[str, Any]:
        """
        Main retraining workflow
        
        Args:
            data_path: Path to training data
            auto_deploy: Automatically deploy if better
            
        Returns:
            Dictionary with retraining results
        """
        try:
            print("=" * 80)
            print("MODEL RETRAINING")
            print("=" * 80)
            
            # Load data
            print("\n[1/6] Loading training data...")
            df = pd.read_csv(data_path)
            logger.info(f"[DATA] Loaded {len(df)} records from {data_path}")
            
            # Prepare features
            print("[2/6] Preparing features...")
            X, y = self.prepare_features(df)
            
            if X is None or y is None:
                return {
                    "success": False,
                    "message": "Failed to prepare features"
                }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Train new model
            print("[3/6] Training new model...")
            new_model = self.train_model(X_train, y_train)
            
            # Validate new model
            print("[4/6] Validating new model...")
            new_metrics = self.validate_model(new_model, X_test, y_test)
            
            # Load and validate current model
            print("[5/6] Comparing with current model...")
            current_metrics = None
            if self.load_current_model():
                current_metrics = self.validate_model(self.current_model, X_test, y_test)
            
            # Compare models
            is_better = self.compare_models(new_metrics, current_metrics)
            
            # Save new model
            print("[6/6] Saving new model...")
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.save_model(new_model, version, new_metrics)
            
            # Deploy if better and auto_deploy enabled
            if is_better and auto_deploy:
                print("\n[DEPLOY] Deploying new model...")
                self.deploy_model(model_path)
                deployed = True
            elif is_better:
                print(f"\n[INFO] New model is better but auto-deploy is disabled")
                print(f"[INFO] To deploy manually, run:")
                print(f"       python scripts/model_retrainer.py --deploy {model_path}")
                deployed = False
            else:
                print(f"\n[INFO] New model not deployed (insufficient improvement)")
                deployed = False
            
            print("\n" + "=" * 80)
            
            return {
                "success": True,
                "message": "Retraining complete",
                "is_better": is_better,
                "deployed": deployed,
                "new_metrics": new_metrics,
                "current_metrics": current_metrics,
                "model_path": model_path,
                "version": version
            }
            
        except Exception as e:
            logger.error("[ERROR] Retraining failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "message": f"Retraining failed: {str(e)}"
            }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Model Retrainer")
    parser.add_argument("--data", type=str, 
                       default="data/training/aggregated_data.csv",
                       help="Path to training data")
    parser.add_argument("--auto-deploy", action="store_true",
                       help="Automatically deploy if model is better")
    parser.add_argument("--min-improvement", type=float, default=0.02,
                       help="Minimum accuracy improvement required (default: 0.02)")
    parser.add_argument("--deploy", type=str,
                       help="Deploy a specific model version")
    
    args = parser.parse_args()
    
    retrainer = ModelRetrainer(min_improvement=args.min_improvement)
    
    # Deploy specific model if requested
    if args.deploy:
        print(f"[DEPLOY] Deploying model: {args.deploy}")
        retrainer.deploy_model(args.deploy)
        return
    
    # Run retraining
    result = retrainer.retrain(
        data_path=args.data,
        auto_deploy=args.auto_deploy
    )
    
    if result['success']:
        print(f"\n[SUCCESS] {result['message']}")
    else:
        print(f"\n[FAILED] {result['message']}")


if __name__ == "__main__":
    main()
