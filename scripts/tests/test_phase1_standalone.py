"""
Phase 1 AI Enhancements - Standalone Test
Tests Model Explainer service independently
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sys
from pathlib import Path
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Now we can import from services
from services.explainer import ModelExplainer

# Define feature columns
FEATURE_COLS = [
    'temp_c', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm', 'cloud', 'wave_height',
    'heat_index', 'wind_pressure_ratio', 'precip_intensity', 'weather_stability',
    'precip_trend_3h', 'temp_change_rate', 'humidity_trend', 'pressure_trend', 'wind_gust_factor',
    'coastal_distance', 'urban_density_proxy', 'elevation_proxy'
]

def test_model_explainer_with_random_forest():
    """Test SHAP explainer with Random Forest"""
    print("\n" + "="*60)
    print("TEST 1: Model Explainer with Random Forest")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(200, len(FEATURE_COLS)),
        columns=FEATURE_COLS
    )
    # High precipitation = high risk
    y = (X['precip_mm'] > 0.5).astype(int)
    
    # Train Random Forest
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    print(f"[OK] Model trained (accuracy on training: {model.score(X, y):.2%})")
    
    # Create explainer
    print("\nInitializing SHAP explainer...")
    explainer = ModelExplainer(model, FEATURE_COLS)
    print("[OK] Explainer initialized")
    
    # Test explanation
    print("\nGenerating explanation for sample prediction...")
    test_sample = X.iloc[[0]]
    prediction = model.predict_proba(test_sample)[0, 1]
    explanation = explainer.explain_prediction(test_sample, top_n=5)
    
    print(f"\n[STATS] Prediction Results:")
    print(f"   Risk Score: {prediction:.1%}")
    print(f"   Explanation Available: {explanation['explanation_available']}")
    
    print(f"\n[INFO] Top 5 Contributing Features:")
    for i, contrib in enumerate(explanation['top_contributors'], 1):
        print(f"   {i}. {contrib['feature']}: {contrib['contribution']:+.1f}% (value: {contrib['value']:.2f})")
    
    print(f"\n[MSG] Interpretation:")
    print(f"   {explanation['interpretation']}")
    
    assert explanation['explanation_available'] == True
    assert len(explanation['top_contributors']) == 5
    print("\n[OK] Random Forest test PASSED")
    return True


def test_model_explainer_with_xgboost():
    """Test SHAP explainer with XGBoost"""
    print("\n" + "="*60)
    print("TEST 2: Model Explainer with XGBoost")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(200, len(FEATURE_COLS)),
        columns=FEATURE_COLS
    )
    y = (X['precip_mm'] > 0.5).astype(int)
    
    # Train XGBoost
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X, y)
    print(f"[OK] XGBoost model trained (accuracy: {model.score(X, y):.2%})")
    
    # Create explainer
    print("\nInitializing SHAP explainer for XGBoost...")
    explainer = ModelExplainer(model, FEATURE_COLS)
    print("[OK] Explainer initialized")
    
    # Test explanation
    print("\nGenerating explanation...")
    test_sample = X.iloc[[0]]
    prediction = model.predict_proba(test_sample)[0, 1]
    explanation = explainer.explain_prediction(test_sample, top_n=5)
    
    print(f"\n[STATS] Prediction Results:")
    print(f"   Risk Score: {prediction:.1%}")
    print(f"   Explanation Available: {explanation['explanation_available']}")
    
    print(f"\n[INFO] Top 5 Contributing Features:")
    for i, contrib in enumerate(explanation['top_contributors'], 1):
        print(f"   {i}. {contrib['feature']}: {contrib['contribution']:+.1f}% (value: {contrib['value']:.2f})")
    
    print(f"\n[MSG] Interpretation:")
    print(f"   {explanation['interpretation']}")
    
    assert explanation['explanation_available'] == True
    assert len(explanation['top_contributors']) == 5
    print("\n[OK] XGBoost test PASSED")
    return True


def test_batch_explanations():
    """Test batch explanation generation"""
    print("\n" + "="*60)
    print("TEST 3: Batch Explanations")
    print("="*60)
    
    # Create data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, len(FEATURE_COLS)),
        columns=FEATURE_COLS
    )
    y = (X['precip_mm'] > 0.5).astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)
    
    # Create explainer
    explainer = ModelExplainer(model, FEATURE_COLS)
    
    # Test batch
    print("Generating explanations for 10 samples...")
    batch_samples = X.iloc[:10]
    explanations = explainer.explain_batch(batch_samples, top_n=3)
    
    print(f"[OK] Generated {len(explanations)} explanations")
    
    # Show first 3
    for i, exp in enumerate(explanations[:3], 1):
        print(f"\n   Sample {i}:")
        print(f"   Top contributor: {exp['top_contributors'][0]['feature']}")
        print(f"   Contribution: {exp['top_contributors'][0]['contribution']:+.1f}%")
    
    assert len(explanations) == 10
    assert all('top_contributors' in exp for exp in explanations)
    print("\n[OK] Batch explanation test PASSED")
    return True


def test_high_risk_scenario():
    """Test explanation for high-risk flood scenario"""
    print("\n" + "="*60)
    print("TEST 4: High-Risk Flood Scenario")
    print("="*60)
    
    # Create training data
    np.random.seed(42)
    X_train = pd.DataFrame(
        np.random.randn(200, len(FEATURE_COLS)),
        columns=FEATURE_COLS
    )
    y_train = (X_train['precip_mm'] > 0.5).astype(int)
    
    # Train model
    model = XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Create high-risk scenario
    high_risk_scenario = pd.DataFrame([{
        'temp_c': 25.0,
        'humidity': 95.0,
        'wind_kph': 70.0,
        'pressure_mb': 990.0,
        'precip_mm': 250.0,  # Extreme rainfall
        'cloud': 100.0,
        'wave_height': 4.0,
        'heat_index': 30.0,
        'wind_pressure_ratio': 0.07,
        'precip_intensity': 25.0,
        'weather_stability': 20000.0,
        'precip_trend_3h': 3.5,
        'temp_change_rate': 1.0,
        'humidity_trend': 10.0,
        'pressure_trend': -5.0,
        'wind_gust_factor': 2.0,
        'coastal_distance': 0.1,
        'urban_density_proxy': 0.9,
        'elevation_proxy': 50.0
    }])
    
    print("Analyzing extreme flood scenario:")
    print(f"   Precipitation: 250mm (extreme)")
    print(f"   Humidity: 95% (very high)")
    print(f"   Wind: 70 km/h (strong)")
    print(f"   Pressure: 990mb (low)")
    
    # Get prediction and explanation
    explainer = ModelExplainer(model, FEATURE_COLS)
    prediction = model.predict_proba(high_risk_scenario)[0, 1]
    explanation = explainer.explain_prediction(high_risk_scenario, top_n=5)
    
    print(f"\n[STATS] Risk Assessment:")
    print(f"   Flood Risk: {prediction:.1%}")
    
    print(f"\n[INFO] Why is risk high?")
    for i, contrib in enumerate(explanation['top_contributors'], 1):
        print(f"   {i}. {contrib['feature']}: {contrib['contribution']:+.1f}% (value: {contrib['value']:.1f})")
    
    print(f"\n[MSG] {explanation['interpretation']}")
    
    print("\n[OK] High-risk scenario test PASSED")
    return True


def run_all_tests():
    """Run all Phase 1 tests"""
    print("\n" + "="*70)
    print(" "*15 + "PHASE 1 AI ENHANCEMENTS - TEST SUITE")
    print("="*70)
    
    results = []
    
    try:
        results.append(("Model Explainer (RF)", test_model_explainer_with_random_forest()))
    except Exception as e:
        print(f"\n[FAIL] Test 1 FAILED: {e}")
        results.append(("Model Explainer (RF)", False))
    
    try:
        results.append(("Model Explainer (XGBoost)", test_model_explainer_with_xgboost()))
    except Exception as e:
        print(f"\n[FAIL] Test 2 FAILED: {e}")
        results.append(("Model Explainer (XGBoost)", False))
    
    try:
        results.append(("Batch Explanations", test_batch_explanations()))
    except Exception as e:
        print(f"\n[FAIL] Test 3 FAILED: {e}")
        results.append(("Batch Explanations", False))
    
    try:
        results.append(("High-Risk Scenario", test_high_risk_scenario()))
    except Exception as e:
        print(f"\n[FAIL] Test 4 FAILED: {e}")
        results.append(("High-Risk Scenario", False))
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[OK] PASSED" if result else "[FAIL] FAILED"
        print(f"   {test_name:.<50} {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! Phase 1 implementation is working correctly.")
    else:
        print(f"\n[WARN]  {total - passed} test(s) failed. Please review the errors above.")
    
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

