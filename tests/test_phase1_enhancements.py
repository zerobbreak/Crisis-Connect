import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import from services
try:
    from services.explainer import ModelExplainer, create_explainer
    from services.predict import FEATURE_COLS
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    raise


class TestModelExplainer:
    """Test suite for SHAP-based model explainer"""
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple trained model for testing"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create synthetic training data
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(100, len(FEATURE_COLS)),
            columns=FEATURE_COLS
        )
        y = (X['precip_mm'] > 0.5).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature data for testing"""
        return pd.DataFrame([{
            'temp_c': 25.0,
            'humidity': 85.0,
            'wind_kph': 45.0,
            'pressure_mb': 1005.0,
            'precip_mm': 120.0,
            'cloud': 95.0,
            'wave_height': 2.5,
            'heat_index': 28.0,
            'wind_pressure_ratio': 0.045,
            'precip_intensity': 15.0,
            'weather_stability': 25000.0,
            'precip_trend_3h': 2.0,
            'temp_change_rate': 0.5,
            'humidity_trend': 5.0,
            'pressure_trend': -3.0,
            'wind_gust_factor': 1.5,
            'coastal_distance': 0.2,
            'urban_density_proxy': 0.8,
            'elevation_proxy': 100.0
        }])
    
    def test_explainer_initialization(self, sample_model):
        """Test that explainer initializes correctly"""
        explainer = ModelExplainer(sample_model, FEATURE_COLS)
        
        assert explainer.model is not None
        assert explainer.feature_names == FEATURE_COLS
        assert explainer.explainer is not None
    
    def test_explain_prediction(self, sample_model, sample_features):
        """Test single prediction explanation"""
        explainer = ModelExplainer(sample_model, FEATURE_COLS)
        explanation = explainer.explain_prediction(sample_features)
        
        # Check explanation structure
        assert 'top_contributors' in explanation
        assert 'all_contributions' in explanation
        assert 'interpretation' in explanation
        assert 'explanation_available' in explanation
        
        # Check top contributors
        assert len(explanation['top_contributors']) <= 5
        assert len(explanation['top_contributors']) > 0
        
        # Check contributor structure
        contributor = explanation['top_contributors'][0]
        assert 'feature' in contributor
        assert 'contribution' in contributor
        assert 'value' in contributor
        assert 'direction' in contributor
    
    def test_explanation_values(self, sample_model, sample_features):
        """Test that explanation values are reasonable"""
        explainer = ModelExplainer(sample_model, FEATURE_COLS)
        explanation = explainer.explain_prediction(sample_features)
        
        # Contributions should be numeric
        for contrib in explanation['top_contributors']:
            assert isinstance(contrib['contribution'], (int, float))
            assert isinstance(contrib['value'], (int, float))
        
        # Direction should be valid
        for contrib in explanation['top_contributors']:
            assert contrib['direction'] in ['increases', 'decreases']
    
    def test_interpretation_generation(self, sample_model, sample_features):
        """Test human-readable interpretation"""
        explainer = ModelExplainer(sample_model, FEATURE_COLS)
        explanation = explainer.explain_prediction(sample_features)
        
        interpretation = explanation['interpretation']
        
        # Should be a non-empty string
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        
        # Should contain readable text
        assert any(word in interpretation.lower() for word in ['risk', 'due', 'contributing'])
    
    def test_batch_explanation(self, sample_model):
        """Test batch explanation for multiple predictions"""
        explainer = ModelExplainer(sample_model, FEATURE_COLS)
        
        # Create batch of features
        batch_features = pd.DataFrame(
            np.random.randn(5, len(FEATURE_COLS)),
            columns=FEATURE_COLS
        )
        
        explanations = explainer.explain_batch(batch_features)
        
        # Should return list of explanations
        assert isinstance(explanations, list)
        assert len(explanations) == 5
        
        # Each should be a valid explanation
        for exp in explanations:
            assert 'top_contributors' in exp
            assert 'interpretation' in exp
    
    def test_global_feature_importance(self, sample_model):
        """Test global feature importance extraction"""
        explainer = ModelExplainer(sample_model, FEATURE_COLS)
        importance = explainer.get_global_feature_importance()
        
        # Should return dictionary
        assert isinstance(importance, dict)
        
        # Should have feature names as keys
        if importance:  # May be empty for some model types
            assert all(feat in FEATURE_COLS for feat in importance.keys())
            assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_fallback_explanation(self, sample_features):
        """Test fallback explanation when SHAP fails"""
        # Create explainer with None model to trigger fallback
        explainer = ModelExplainer(None, FEATURE_COLS)
        explainer.explainer = None  # Force fallback
        
        explanation = explainer._fallback_explanation(sample_features)
        
        # Should still return valid structure
        assert 'top_contributors' in explanation
        assert 'interpretation' in explanation
        assert explanation['explanation_available'] == False
    
    def test_feature_name_humanization(self, sample_model):
        """Test feature name conversion to human-readable format"""
        explainer = ModelExplainer(sample_model, FEATURE_COLS)
        
        # Test various feature names
        assert 'precipitation' in explainer._humanize_feature_name('precip_mm')
        assert 'temperature' in explainer._humanize_feature_name('temp_c')
        assert 'wind speed' in explainer._humanize_feature_name('wind_kph')
        assert 'humidity' in explainer._humanize_feature_name('humidity')


class TestXGBoostIntegration:
    """Test XGBoost model integration"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        data = pd.DataFrame(
            np.random.randn(200, len(FEATURE_COLS)),
            columns=FEATURE_COLS
        )
        # Create target based on precipitation
        data['is_severe'] = (data['precip_mm'] > 0.5).astype(int)
        return data
    
    def test_xgboost_import(self):
        """Test that XGBoost can be imported"""
        try:
            from xgboost import XGBClassifier
            assert True
        except ImportError:
            pytest.fail("XGBoost not installed")
    
    def test_xgboost_training(self, sample_data):
        """Test XGBoost model training"""
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        
        X = sample_data[FEATURE_COLS]
        y = sample_data['is_severe']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Test prediction
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_xgboost_probability(self, sample_data):
        """Test XGBoost probability predictions"""
        from xgboost import XGBClassifier
        
        X = sample_data[FEATURE_COLS]
        y = sample_data['is_severe']
        
        model = XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
        model.fit(X, y)
        
        probas = model.predict_proba(X[:10])
        
        # Should return probabilities
        assert probas.shape == (10, 2)
        assert all(0 <= p <= 1 for row in probas for p in row)
        assert all(abs(sum(row) - 1.0) < 0.01 for row in probas)
    
    def test_xgboost_with_explainer(self, sample_data):
        """Test XGBoost model with SHAP explainer"""
        from xgboost import XGBClassifier
        
        X = sample_data[FEATURE_COLS]
        y = sample_data['is_severe']
        
        model = XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
        model.fit(X, y)
        
        # Create explainer
        explainer = ModelExplainer(model, FEATURE_COLS)
        
        # Test explanation
        explanation = explainer.explain_prediction(X.iloc[[0]])
        
        assert explanation['explanation_available'] == True
        assert len(explanation['top_contributors']) > 0


class TestEnhancedFeatures:
    """Test enhanced feature engineering"""
    
    def test_interaction_features(self):
        """Test interaction feature creation"""
        features = pd.DataFrame([{
            'temp_c': 25.0,
            'humidity': 80.0,
            'wind_kph': 40.0,
            'precip_mm': 50.0
        }])
        
        # Add interaction features
        features['temp_humidity_interaction'] = features['temp_c'] * features['humidity'] / 100
        features['wind_precip_interaction'] = features['wind_kph'] * features['precip_mm']
        
        assert 'temp_humidity_interaction' in features.columns
        assert 'wind_precip_interaction' in features.columns
        assert features['temp_humidity_interaction'].iloc[0] == 20.0  # 25 * 80 / 100
        assert features['wind_precip_interaction'].iloc[0] == 2000.0  # 40 * 50
    
    def test_extreme_indicators(self):
        """Test extreme event indicator creation"""
        features = pd.DataFrame([{
            'precip_mm': 120.0,
            'wind_kph': 85.0,
            'pressure_mb': 995.0
        }])
        
        # Add extreme indicators
        features['is_extreme_rain'] = (features['precip_mm'] > 100).astype(int)
        features['is_extreme_wind'] = (features['wind_kph'] > 80).astype(int)
        features['is_low_pressure'] = (features['pressure_mb'] < 1000).astype(int)
        
        assert features['is_extreme_rain'].iloc[0] == 1
        assert features['is_extreme_wind'].iloc[0] == 1
        assert features['is_low_pressure'].iloc[0] == 1


class TestUncertaintyQuantification:
    """Test uncertainty quantification methods"""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing"""
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(100, len(FEATURE_COLS)),
            columns=FEATURE_COLS
        )
        y = (X['precip_mm'] > 0.5).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model, X
    
    def test_bootstrap_uncertainty(self, trained_model):
        """Test bootstrap-based uncertainty calculation"""
        model, X = trained_model
        
        # Simple bootstrap implementation
        n_iterations = 50
        predictions = []
        
        for _ in range(n_iterations):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X.iloc[indices]
            pred = model.predict_proba(X_boot)[:, 1]
            predictions.append(pred[:10])  # First 10 predictions
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        lower_95 = np.percentile(predictions, 2.5, axis=0)
        upper_95 = np.percentile(predictions, 97.5, axis=0)
        
        # Validate results
        assert len(mean_pred) == 10
        assert all(0 <= p <= 1 for p in mean_pred)
        assert all(s >= 0 for s in std_pred)
        assert all(lower_95 <= upper_95)


def test_phase1_integration():
    """Integration test for all Phase 1 components"""
    from sklearn.ensemble import RandomForestClassifier
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, len(FEATURE_COLS)),
        columns=FEATURE_COLS
    )
    y = (X['precip_mm'] > 0.5).astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = ModelExplainer(model, FEATURE_COLS)
    
    # Make prediction with explanation
    test_sample = X.iloc[[0]]
    prediction = model.predict_proba(test_sample)[0, 1]
    explanation = explainer.explain_prediction(test_sample)
    
    # Validate integration
    assert 0 <= prediction <= 1
    assert explanation['explanation_available'] == True
    assert len(explanation['top_contributors']) > 0
    
    print("\n✅ Phase 1 Integration Test Passed!")
    print(f"   Prediction: {prediction:.2%}")
    print(f"   Top contributor: {explanation['top_contributors'][0]['feature']}")
    print(f"   Interpretation: {explanation['interpretation'][:100]}...")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
