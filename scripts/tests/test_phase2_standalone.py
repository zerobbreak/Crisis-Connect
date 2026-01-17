#!/usr/bin/env python3
"""
Phase 2 AI Enhancements - Standalone Test Suite
Tests LSTM-based time series forecasting implementation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

print("=" * 70)
print("           PHASE 2 AI ENHANCEMENTS - TEST SUITE")
print("=" * 70)
print()


def test_time_series_data_service():
    """Test 1: Time Series Data Service"""
    print("=" * 60)
    print("TEST 1: Time Series Data Service")
    print("=" * 60)
    
    try:
        from services.time_series_data import TimeSeriesDataService
        
        # Initialize service
        print("Initializing TimeSeriesDataService...")
        service = TimeSeriesDataService(sequence_length=24, forecast_horizons=[24, 48, 72])
        print(f"[OK] Service initialized: seq_len={service.sequence_length}")
        
        # Test historical data fetching
        print("\nFetching historical weather data for Durban...")
        df = service.fetch_historical_weather(
            lat=-29.8587,
            lon=31.0218,
            days_back=30,  # Just 30 days for testing
            location_name="Durban"
        )
        
        print(f"[OK] Fetched {len(df)} hours of data")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Test sequence generation
        print("\nGenerating sequences...")
        X, y = service.create_sequences(df)
        print(f"[OK] Created {len(X)} sequences")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        # Test normalization
        print("\nNormalizing data...")
        X_normalized = service.normalize_data(X, fit=True)
        print(f"[OK] Data normalized")
        print(f"   Min: {X_normalized.min():.4f}, Max: {X_normalized.max():.4f}")
        
        print("\n[OK] Time Series Data Service test PASSED")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Time Series Data Service test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lstm_model():
    """Test 2: LSTM Model Architecture"""
    print("\n" + "=" * 60)
    print("TEST 2: LSTM Model Architecture")
    print("=" * 60)
    
    try:
        from services.lstm_model import create_lstm_model
        
        # Create model
        print("Building LSTM model...")
        model = create_lstm_model(
            sequence_length=24,
            n_features=11,
            n_horizons=3,
            lstm_units=[64, 32],
            dense_units=[16, 8]
        )
        
        model.compile_model(learning_rate=0.001)
        print("[OK] Model built and compiled")
        
        # Test prediction with dummy data
        print("\nTesting prediction with dummy data...")
        X_dummy = np.random.rand(5, 24, 11)  # 5 samples, 24 timesteps, 11 features
        predictions = model.predict(X_dummy)
        
        print(f"[OK] Predictions generated")
        print(f"   Input shape: {X_dummy.shape}")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Sample predictions: {predictions[0]}")
        
        # Test single prediction
        print("\nTesting single prediction...")
        X_single = np.random.rand(1, 24, 11)
        risk_scores, confidence = model.predict_single(X_single)
        
        print(f"[OK] Single prediction generated")
        print(f"   Risk scores: {risk_scores}")
        print(f"   Confidence: {confidence}")
        
        print("\n[OK] LSTM Model test PASSED")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] LSTM Model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forecast_service():
    """Test 3: Forecast Service"""
    print("\n" + "=" * 60)
    print("TEST 3: Forecast Service")
    print("=" * 60)
    
    try:
        from services.forecast import ForecastService
        
        # Initialize service (will work without model)
        print("Initializing ForecastService...")
        service = ForecastService(
            model_path="data/models/lstm_forecast.h5",
            sequence_length=24,
            forecast_horizons=[24, 48, 72]
        )
        print("[OK] Service initialized")
        
        # Test forecast generation (will use fallback if no model)
        print("\nGenerating forecast for Durban...")
        forecast = service.generate_forecast(
            location_id="test_001",
            location_name="Durban",
            lat=-29.8587,
            lon=31.0218
        )
        
        print(f"[OK] Forecast generated")
        print(f"   Location: {forecast['location_name']}")
        print(f"   Current Risk: {forecast['current_risk']}")
        print(f"   Forecast Available: {forecast['forecast_available']}")
        
        if forecast['forecast_available']:
            print(f"   Number of forecasts: {len(forecast['forecasts'])}")
            for f in forecast['forecasts']:
                print(f"      {f['horizon_hours']}h: {f['predicted_risk']}% ({f['trend']})")
        else:
            print("   [INFO] LSTM model not available, using fallback")
        
        print("\n[OK] Forecast Service test PASSED")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Forecast Service test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training():
    """Test 4: Mini Training Run"""
    print("\n" + "=" * 60)
    print("TEST 4: Mini Training Run (5 epochs)")
    print("=" * 60)
    
    try:
        from services.time_series_data import TimeSeriesDataService
        from services.lstm_model import create_lstm_model
        
        # Prepare small dataset
        print("Preparing small training dataset...")
        data_service = TimeSeriesDataService(sequence_length=24, forecast_horizons=[24, 48, 72])
        
        # Fetch data for just one location
        df = data_service.fetch_historical_weather(
            lat=-29.8587,
            lon=31.0218,
            days_back=60,  # 2 months
            location_name="Durban"
        )
        
        X, y = data_service.create_sequences(df)
        X_normalized = data_service.normalize_data(X, fit=True)
        
        # Split data
        split_idx = int(len(X_normalized) * 0.8)
        X_train, X_val = X_normalized[:split_idx], X_normalized[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"[OK] Dataset prepared: {len(X_train)} train, {len(X_val)} val samples")
        
        # Build small model
        print("\nBuilding and training small LSTM model...")
        model = create_lstm_model(
            sequence_length=24,
            n_features=X_train.shape[2],
            n_horizons=y_train.shape[1],
            lstm_units=[32, 16],  # Smaller for faster training
            dense_units=[8]
        )
        
        model.compile_model(learning_rate=0.001)
        
        # Train for just 5 epochs
        print("Training for 5 epochs (this will take 1-2 minutes)...")
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=5,
            batch_size=16,
            early_stopping_patience=3,
            verbose=0  # Quiet mode
        )
        
        print(f"[OK] Training completed")
        print(f"   Final training loss: {history['loss'][-1]:.4f}")
        print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
        
        # Test prediction
        print("\nTesting prediction on validation data...")
        predictions = model.predict(X_val[:5])
        print(f"[OK] Predictions generated")
        print(f"   Sample predictions: {predictions[0]}")
        
        print("\n[OK] Mini Training test PASSED")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] Mini Training test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    results = {
        "Time Series Data Service": test_time_series_data_service(),
        "LSTM Model Architecture": test_lstm_model(),
        "Forecast Service": test_forecast_service(),
        "Mini Training Run": test_mini_training()
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("                     TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[FAILED]"
        print(f"   {test_name:.<50} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n   Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! Phase 2 implementation is working correctly.")
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please review the errors above.")
    
    print("=" * 70)
    print()
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
