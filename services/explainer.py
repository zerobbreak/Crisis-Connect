"""
Model Explainer Service - SHAP-based Explainable AI
Provides interpretable explanations for flood risk predictions
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("crisisconnect.explainer")

class ModelExplainer:
    """
    Provides SHAP-based explanations for model predictions.
    Shows which features contribute most to flood risk scores.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer with trained model.
        
        Args:
            model: Trained XGBoost or RandomForest model
            feature_names: List of feature names used in model
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            # TreeExplainer works for tree-based models (XGBoost, RF)
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("✅ SHAP TreeExplainer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize TreeExplainer: {e}")
            # Fallback to KernelExplainer (slower but works for any model)
            try:
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(pd.DataFrame(columns=self.feature_names), 100)
                )
                logger.info("✅ SHAP KernelExplainer initialized (fallback)")
            except Exception as e2:
                logger.error(f"Failed to initialize any explainer: {e2}")
                self.explainer = None
    
    def explain_prediction(self, features: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            features: DataFrame with feature values (single row)
            top_n: Number of top contributing features to return

        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            return self._fallback_explanation(features)

        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification returns list of arrays
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            shap_values = np.ravel(np.array(shap_values))  # flatten to 1D

            # Create feature importance dictionary
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                if i < len(shap_values):
                    feature_importance[feature_name] = float(shap_values[i])

            # Sort by absolute contribution
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Get top contributors
            top_contributors = []
            for feature, contribution in sorted_features[:top_n]:
                feature_value = features[feature].iloc[0] if feature in features.columns else 0
                top_contributors.append({
                    'feature': feature,
                    'contribution': round(contribution * 100, 2),  # Scale to percentage
                    'value': round(float(feature_value), 2),
                    'direction': 'increases' if contribution > 0 else 'decreases'
                })

            # Generate human-readable interpretation
            interpretation = self._generate_interpretation(top_contributors)

            # Get base value (expected value)
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = np.ravel(np.array(base_value))
                base_value = float(base_value[-1])  # use positive class

            return {
                'top_contributors': top_contributors,
                'all_contributions': {k: round(v * 100, 2) for k, v in feature_importance.items()},
                'base_risk': round(base_value * 100, 2),
                'interpretation': interpretation,
                'explanation_available': True
            }

        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return self._fallback_explanation(features)

    
    def _generate_interpretation(self, top_contributors: List[Dict]) -> str:
        """Generate human-readable interpretation of prediction"""
        if not top_contributors:
            return "Unable to generate explanation"
        
        # Get top positive and negative contributors
        positive = [c for c in top_contributors if c['contribution'] > 0]
        negative = [c for c in top_contributors if c['contribution'] < 0]
        
        interpretation_parts = []
        
        if positive:
            top_positive = positive[0]
            feature_name = self._humanize_feature_name(top_positive['feature'])
            interpretation_parts.append(
                f"High risk primarily due to {feature_name} "
                f"({top_positive['value']}) contributing +{abs(top_positive['contribution'])}%"
            )
        
        if len(positive) > 1:
            second_positive = positive[1]
            feature_name = self._humanize_feature_name(second_positive['feature'])
            interpretation_parts.append(
                f"Also elevated by {feature_name} "
                f"({second_positive['value']}) adding +{abs(second_positive['contribution'])}%"
            )
        
        if negative:
            top_negative = negative[0]
            feature_name = self._humanize_feature_name(top_negative['feature'])
            interpretation_parts.append(
                f"Risk reduced by {feature_name} "
                f"({top_negative['value']}) lowering by {abs(top_negative['contribution'])}%"
            )
        
        return ". ".join(interpretation_parts) + "."
    
    def _humanize_feature_name(self, feature: str) -> str:
        """Convert feature names to human-readable format"""
        name_mapping = {
            'precip_mm': 'precipitation',
            'temp_c': 'temperature',
            'wind_kph': 'wind speed',
            'pressure_mb': 'atmospheric pressure',
            'humidity': 'humidity',
            'wave_height': 'wave height',
            'heat_index': 'heat index',
            'precip_intensity': 'rainfall intensity',
            'precip_trend_3h': 'precipitation trend',
            'pressure_trend': 'pressure change',
            'humidity_trend': 'humidity change',
            'temp_change_rate': 'temperature change',
            'wind_pressure_ratio': 'wind-pressure ratio',
            'weather_stability': 'weather stability',
            'coastal_distance': 'distance from coast',
            'urban_density_proxy': 'urban density',
            'elevation_proxy': 'elevation',
            'wind_gust_factor': 'wind gusts'
        }
        return name_mapping.get(feature, feature.replace('_', ' '))
    
    def _fallback_explanation(self, features: pd.DataFrame) -> Dict:
        """Provide basic explanation when SHAP is unavailable"""
        # Simple rule-based explanation
        top_contributors = []
        
        # Check key features
        if 'precip_mm' in features.columns:
            precip = features['precip_mm'].iloc[0]
            if precip > 50:
                top_contributors.append({
                    'feature': 'precip_mm',
                    'contribution': min(precip / 3, 50),
                    'value': precip,
                    'direction': 'increases'
                })
        
        if 'humidity' in features.columns:
            humidity = features['humidity'].iloc[0]
            if humidity > 80:
                top_contributors.append({
                    'feature': 'humidity',
                    'contribution': (humidity - 80) / 2,
                    'value': humidity,
                    'direction': 'increases'
                })
        
        return {
            'top_contributors': top_contributors[:5],
            'all_contributions': {},
            'base_risk': 50.0,
            'interpretation': "Explanation based on rule-based analysis (SHAP unavailable)",
            'explanation_available': False
        }
    
    def explain_batch(self, features: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            features: DataFrame with multiple rows
            top_n: Number of top features per prediction
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        for idx in range(len(features)):
            single_row = features.iloc[[idx]]
            explanation = self.explain_prediction(single_row, top_n)
            explanations.append(explanation)
        
        return explanations
    
    def get_global_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance across all predictions.
        
        Returns:
            Dictionary of feature names to importance scores
        """
        if self.explainer is None:
            return {}
        
        try:
            # For tree-based models, use built-in feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                return dict(zip(self.feature_names, importance))
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting global feature importance: {e}")
            return {}


def create_explainer(model, feature_names: List[str]) -> ModelExplainer:
    """
    Factory function to create model explainer.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        ModelExplainer instance
    """
    return ModelExplainer(model, feature_names)
