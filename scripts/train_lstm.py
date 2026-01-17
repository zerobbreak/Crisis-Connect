#!/usr/bin/env python3
"""
LSTM Model Training Script
Trains LSTM model for flood risk forecasting using historical weather data
"""

import sys
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.time_series_data import TimeSeriesDataService
from services.lstm_model import create_lstm_model
from services.predict import DISTRICT_COORDS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train LSTM Flood Forecasting Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--sequence-length", type=int, default=24, help="Input sequence length (hours)")
    parser.add_argument("--days-back", type=int, default=730, help="Days of historical data to fetch")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="data/models/lstm_forecast.h5", help="Output model path")
    parser.add_argument("--locations", type=int, default=5, help="Number of locations to use for training")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LSTM FLOOD FORECASTING MODEL - TRAINING SCRIPT")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Sequence Length: {args.sequence_length} hours")
    print(f"  Historical Data: {args.days_back} days")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Output Model: {args.output}")
    print("=" * 80)
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Initialize data service
    print("\n[1/5] Initializing Time Series Data Service...")
    data_service = TimeSeriesDataService(
        sequence_length=args.sequence_length,
        forecast_horizons=[24, 48, 72]
    )
    print("✅ Data service initialized")
    
    # Step 2: Prepare training data
    print(f"\n[2/5] Fetching historical data for {args.locations} locations...")
    print("This may take several minutes...")
    
    # Select subset of locations for training
    selected_locations = dict(list(DISTRICT_COORDS.items())[:args.locations])
    
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = data_service.prepare_training_data(
            locations=selected_locations,
            days_back=args.days_back,
            train_split=0.7,
            val_split=0.15
        )
        
        print(f"\n✅ Training data prepared:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Input shape: {X_train.shape}")
        print(f"   Output shape: {y_train.shape}")
        
    except Exception as e:
        print(f"\n❌ Failed to prepare training data: {e}")
        print("Please check your internet connection and try again.")
        return 1
    
    # Step 3: Build LSTM model
    print("\n[3/5] Building LSTM Model...")
    n_features = X_train.shape[2]
    n_horizons = y_train.shape[1]
    
    model = create_lstm_model(
        sequence_length=args.sequence_length,
        n_features=n_features,
        n_horizons=n_horizons,
        lstm_units=[128, 64],
        dense_units=[32, 16],
        dropout_rate=0.2
    )
    
    model.compile_model(learning_rate=args.learning_rate)
    print("✅ Model built and compiled")
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Step 4: Train model
    print(f"\n[4/5] Training Model ({args.epochs} epochs)...")
    print("This may take 30-60 minutes depending on your hardware...")
    print("-" * 80)
    
    try:
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=10,
            reduce_lr_patience=5,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
        # Print training summary
        final_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        best_val_loss = min(history['val_loss'])
        
        print(f"\nTraining Summary:")
        print(f"  Final Training Loss: {final_loss:.4f}")
        print(f"  Final Validation Loss: {final_val_loss:.4f}")
        print(f"  Best Validation Loss: {best_val_loss:.4f}")
        print(f"  Total Epochs: {len(history['loss'])}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1
    
    # Step 5: Evaluate on test set
    print("\n[5/5] Evaluating on Test Set...")
    try:
        metrics = model.evaluate(X_test, y_test)
        
        print("\n✅ Test Evaluation:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"   {metric.upper()}: {value:.4f}")
        
    except Exception as e:
        print(f"\n⚠️ Evaluation failed: {e}")
    
    # Save model
    print(f"\n💾 Saving model to {args.output}...")
    try:
        model.save(args.output)
        print(f"✅ Model saved successfully!")
        
        # Save training metadata
        metadata = {
            "trained_at": datetime.now().isoformat(),
            "epochs": len(history['loss']),
            "sequence_length": args.sequence_length,
            "n_features": n_features,
            "n_horizons": n_horizons,
            "locations_used": list(selected_locations.keys()),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "final_loss": float(final_loss),
            "final_val_loss": float(final_val_loss),
            "best_val_loss": float(best_val_loss),
            "test_metrics": {k: float(v) if v is not None else None for k, v in metrics.items()}
        }
        
        metadata_path = args.output.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"\n❌ Failed to save model: {e}")
        return 1
    
    # Success
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nModel saved to: {args.output}")
    print(f"Metadata saved to: {metadata_path}")
    print("\nYou can now use the model for forecasting via the API:")
    print(f"  GET /api/v1/forecast/{{location_id}}")
    print("\n" + "=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
