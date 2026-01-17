"""
LSTM Model for Flood Risk Forecasting
Implements LSTM neural network for time series prediction of flood risk
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, models, callbacks
    from keras.models import load_model as keras_load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM forecasting will not work.")

logger = logging.getLogger("crisisconnect.lstm_model")


class LSTMForecastModel:
    """
    LSTM-based model for multi-horizon flood risk forecasting.
    Predicts risk scores at 24h, 48h, and 72h horizons.
    """
    
    def __init__(
        self,
        sequence_length: int = 24,
        n_features: int = 11,
        n_horizons: int = 3,
        lstm_units: List[int] = [128, 64],
        dense_units: List[int] = [32, 16],
        dropout_rate: float = 0.2
    ):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps in input sequence
            n_features: Number of features per time step
            n_horizons: Number of forecast horizons (default: 3 for 24h, 48h, 72h)
            lstm_units: List of LSTM layer units
            dense_units: List of dense layer units
            dropout_rate: Dropout rate for regularization
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow>=2.15.0")
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_horizons = n_horizons
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.history = None
        
        logger.info(f"LSTMForecastModel initialized: seq_len={sequence_length}, features={n_features}, horizons={n_horizons}")
    
    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)  # Return sequences for all but last LSTM
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                name=f'lstm_{i+1}'
            )(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_lstm_{i+1}')(x)
        
        # Dense layers
        for i, units in enumerate(self.dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_dense_{i+1}')(x)
        
        # Output layer - predict risk scores for each horizon
        outputs = layers.Dense(self.n_horizons, activation='sigmoid', name='output')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='lstm_forecast')
        
        logger.info("✅ LSTM model architecture built")
        logger.info(f"   Input shape: (None, {self.sequence_length}, {self.n_features})")
        logger.info(f"   Output shape: (None, {self.n_horizons})")
        
        return model
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        loss: str = 'mse',
        metrics: Optional[List[str]] = None
    ):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
            loss: Loss function (default: 'mse')
            metrics: List of metrics to track
        """
        if self.model is None:
            self.model = self.build_model()
        
        if metrics is None:
            metrics = ['mae', 'mse']
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"✅ Model compiled with lr={learning_rate}, loss={loss}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        verbose: int = 1
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features (n_samples, sequence_length, n_features)
            y_train: Training targets (n_samples, n_horizons)
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.compile_model()
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'data/models/lstm_forecast_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        logger.info(f"🚀 Starting training: {epochs} epochs, batch_size={batch_size}")
        logger.info(f"   Training samples: {len(X_train)}")
        logger.info(f"   Validation samples: {len(X_val)}")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        logger.info("✅ Training completed")
        
        return self.history.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'mae': results[1] if len(results) > 1 else None,
            'mse': results[2] if len(results) > 2 else None
        }
        
        logger.info("📊 Test Evaluation:")
        for metric, value in metrics.items():
            if value is not None:
                logger.info(f"   {metric}: {value:.4f}")
        
        return metrics
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
            batch_size: Batch size for prediction
            
        Returns:
            Predictions (n_samples, n_horizons)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
        return predictions
    
    def predict_single(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate prediction for a single sequence with confidence estimates.
        
        Args:
            X: Single input sequence (1, sequence_length, n_features)
            
        Returns:
            Tuple of (predictions, confidence_dict)
        """
        if X.shape[0] != 1:
            raise ValueError(f"Expected single sample, got {X.shape[0]} samples")
        
        # Get prediction
        pred = self.predict(X)[0]  # Shape: (n_horizons,)
        
        # Convert to risk scores (0-100)
        risk_scores = pred * 100
        
        # Simple confidence estimation (can be improved with uncertainty quantification)
        confidence = {
            '24h': float(risk_scores[0]),
            '48h': float(risk_scores[1]) if len(risk_scores) > 1 else None,
            '72h': float(risk_scores[2]) if len(risk_scores) > 2 else None
        }
        
        return risk_scores, confidence
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(filepath)
        logger.info(f"💾 Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        self.model = keras_load_model(filepath)
        logger.info(f"✅ Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            logger.warning("Model not built yet")
            return
        
        self.model.summary()


def create_lstm_model(
    sequence_length: int = 24,
    n_features: int = 11,
    n_horizons: int = 3,
    **kwargs
) -> LSTMForecastModel:
    """
    Factory function to create LSTM forecast model.
    
    Args:
        sequence_length: Number of time steps in input
        n_features: Number of features per time step
        n_horizons: Number of forecast horizons
        **kwargs: Additional arguments for LSTMForecastModel
        
    Returns:
        LSTMForecastModel instance
    """
    return LSTMForecastModel(
        sequence_length=sequence_length,
        n_features=n_features,
        n_horizons=n_horizons,
        **kwargs
    )


def load_pretrained_model(filepath: str) -> LSTMForecastModel:
    """
    Load a pre-trained LSTM model.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded LSTMForecastModel instance
    """
    model = LSTMForecastModel()
    model.load(filepath)
    return model
