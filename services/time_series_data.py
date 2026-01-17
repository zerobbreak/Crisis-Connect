"""
Time Series Data Preparation Service
Handles historical weather data fetching and sequence generation for LSTM forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import MinMaxScaler
import openmeteo_requests
import requests_cache
from retry_requests import retry

logger = logging.getLogger("crisisconnect.time_series_data")

class TimeSeriesDataService:
    """
    Service for preparing time series data for LSTM model training and inference.
    Handles historical data fetching, sequence generation, and normalization.
    """
    
    def __init__(self, sequence_length: int = 24, forecast_horizons: List[int] = [24, 48, 72]):
        """
        Initialize time series data service.
        
        Args:
            sequence_length: Number of hours of historical data to use as input
            forecast_horizons: List of forecast horizons in hours (e.g., [24, 48, 72])
        """
        self.sequence_length = sequence_length
        self.forecast_horizons = forecast_horizons
        self.scaler = MinMaxScaler()
        
        # Features to use for LSTM input
        self.feature_columns = [
            'temp_c', 'humidity', 'wind_kph', 'pressure_mb', 'precip_mm',
            'cloud', 'wave_height', 'heat_index', 'wind_pressure_ratio',
            'precip_intensity', 'weather_stability'
        ]
        
        logger.info(f"TimeSeriesDataService initialized: seq_len={sequence_length}, horizons={forecast_horizons}")
    
    def fetch_historical_weather(
        self, 
        lat: float, 
        lon: float, 
        days_back: int = 730,
        location_name: str = "Unknown"
    ) -> pd.DataFrame:
        """
        Fetch historical weather data from Open-Meteo Archive API.
        
        Args:
            lat: Latitude
            lon: Longitude
            days_back: Number of days of historical data to fetch (default: 730 = 2 years)
            location_name: Name of location for logging
            
        Returns:
            DataFrame with hourly weather data
        """
        try:
            # Setup Open-Meteo client with caching
            cache_session = requests_cache.CachedSession('.cache', expire_after=86400)  # 24h cache
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            
            url = "https://archive-api.open-meteo.com/v1/archive"
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "hourly": [
                    "temperature_2m", "precipitation", "wind_speed_10m",
                    "relative_humidity_2m", "pressure_msl", "cloud_cover"
                ],
                "timezone": "auto"
            }
            
            logger.info(f"Fetching historical data for {location_name}: {start_date} to {end_date}")
            
            responses = openmeteo.weather_api(url, params=params)
            if not responses:
                raise ValueError("No response from historical API")
            
            response = responses[0]
            hourly = response.Hourly()
            
            # Extract hourly data
            hourly_data = {
                "timestamp": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s"),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                ),
                "temp_c": hourly.Variables(0).ValuesAsNumpy(),
                "precip_mm": hourly.Variables(1).ValuesAsNumpy(),
                "wind_kph": hourly.Variables(2).ValuesAsNumpy() * 3.6,  # m/s to km/h
                "humidity": hourly.Variables(3).ValuesAsNumpy(),
                "pressure_mb": hourly.Variables(4).ValuesAsNumpy(),
                "cloud": hourly.Variables(5).ValuesAsNumpy(),
            }
            
            df = pd.DataFrame(hourly_data)
            
            # Add derived features
            df = self._add_derived_features(df, lat, lon)
            
            # Clean data
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.ffill(inplace=True)  # Forward fill missing values
            df.fillna(0, inplace=True)  # Fill remaining with 0
            
            logger.info(f"✅ Fetched {len(df)} hours of historical data for {location_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {location_name}: {e}")
            raise
    
    def _add_derived_features(self, df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
        """Add derived weather features to the dataframe."""
        # Heat index
        df['heat_index'] = df['temp_c'] + (df['humidity'] / 100) * 5
        
        # Wind-pressure ratio
        df['wind_pressure_ratio'] = df['wind_kph'] / (df['pressure_mb'] / 1000)
        
        # Precipitation intensity (rolling average)
        df['precip_intensity'] = df['precip_mm'].rolling(window=3, min_periods=1).mean()
        
        # Weather stability
        df['weather_stability'] = (df['pressure_mb'] * df['humidity']) / (df['temp_c'] + 273.15)
        
        # Wave height (simplified - only for coastal areas)
        coastal_distance = abs(lon - 30.0)  # Rough approximation
        df['wave_height'] = np.where(coastal_distance < 2.0, 1.0, 0.0)
        
        return df
    
    def create_sequences(
        self, 
        df: pd.DataFrame,
        target_column: str = 'is_severe'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM training.
        
        Args:
            df: DataFrame with weather data
            target_column: Column to use as target (default: 'is_severe')
            
        Returns:
            Tuple of (X, y) where:
                X: shape (n_samples, sequence_length, n_features)
                y: shape (n_samples, n_horizons) - risk scores at each horizon
        """
        # Ensure we have the target column
        if target_column not in df.columns:
            # Create synthetic target based on precipitation and wind
            df['is_severe'] = ((df['precip_mm'] > 50) | 
                              (df['wind_kph'] > 60) | 
                              (df['precip_mm'] > 20) & (df['wind_kph'] > 40)).astype(int)
        
        # Select features
        feature_data = df[self.feature_columns].values
        target_data = df[target_column].values
        
        X_sequences = []
        y_sequences = []
        
        # Create sliding windows
        max_horizon = max(self.forecast_horizons)
        for i in range(len(df) - self.sequence_length - max_horizon):
            # Input: last sequence_length hours
            X_seq = feature_data[i:i + self.sequence_length]
            
            # Output: risk at each forecast horizon
            y_seq = []
            for horizon in self.forecast_horizons:
                future_idx = i + self.sequence_length + horizon - 1
                if future_idx < len(target_data):
                    y_seq.append(target_data[future_idx])
                else:
                    y_seq.append(0)  # Padding
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        logger.info(f"Created {len(X)} sequences: X shape={X.shape}, y shape={y.shape}")
        return X, y
    
    def normalize_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize feature data using MinMaxScaler.
        
        Args:
            X: Input data of shape (n_samples, sequence_length, n_features)
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Normalized data
        """
        n_samples, seq_len, n_features = X.shape
        
        # Reshape to 2D for scaling
        X_reshaped = X.reshape(-1, n_features)
        
        if fit:
            X_scaled = self.scaler.fit_transform(X_reshaped)
            logger.info("✅ Fitted scaler on training data")
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        # Reshape back to 3D
        X_normalized = X_scaled.reshape(n_samples, seq_len, n_features)
        return X_normalized
    
    def prepare_training_data(
        self,
        locations: Dict[str, Tuple[float, float]],
        days_back: int = 730,
        train_split: float = 0.7,
        val_split: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare complete training dataset from multiple locations.
        
        Args:
            locations: Dict of {location_name: (lat, lon)}
            days_back: Days of historical data to fetch
            train_split: Fraction of data for training
            val_split: Fraction of data for validation (rest is test)
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        all_X = []
        all_y = []
        
        for location_name, (lat, lon) in locations.items():
            try:
                # Fetch historical data
                df = self.fetch_historical_weather(lat, lon, days_back, location_name)
                
                # Create sequences
                X, y = self.create_sequences(df)
                
                all_X.append(X)
                all_y.append(y)
                
            except Exception as e:
                logger.warning(f"Failed to prepare data for {location_name}: {e}")
                continue
        
        if not all_X:
            raise ValueError("No data could be prepared from any location")
        
        # Combine all locations
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        
        # Normalize features
        X_normalized = self.normalize_data(X_combined, fit=True)
        
        # Split into train/val/test
        n_samples = len(X_normalized)
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        X_train = X_normalized[:train_end]
        y_train = y_combined[:train_end]
        
        X_val = X_normalized[train_end:val_end]
        y_val = y_combined[train_end:val_end]
        
        X_test = X_normalized[val_end:]
        y_test = y_combined[val_end:]
        
        logger.info(f"✅ Training data prepared:")
        logger.info(f"   Train: {len(X_train)} samples")
        logger.info(f"   Val: {len(X_val)} samples")
        logger.info(f"   Test: {len(X_test)} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def prepare_inference_data(
        self,
        recent_weather: pd.DataFrame
    ) -> np.ndarray:
        """
        Prepare recent weather data for inference.
        
        Args:
            recent_weather: DataFrame with last sequence_length hours of weather data
            
        Returns:
            Normalized sequence ready for LSTM prediction
        """
        # Add derived features if not present
        if 'heat_index' not in recent_weather.columns:
            recent_weather = self._add_derived_features(
                recent_weather, 
                lat=recent_weather.get('lat', [0])[0] if 'lat' in recent_weather.columns else 0,
                lon=recent_weather.get('lon', [0])[0] if 'lon' in recent_weather.columns else 0
            )
        
        # Select features
        feature_data = recent_weather[self.feature_columns].values
        
        # Ensure we have exactly sequence_length hours
        if len(feature_data) < self.sequence_length:
            # Pad with zeros if insufficient data
            padding = np.zeros((self.sequence_length - len(feature_data), len(self.feature_columns)))
            feature_data = np.vstack([padding, feature_data])
        elif len(feature_data) > self.sequence_length:
            # Take last sequence_length hours
            feature_data = feature_data[-self.sequence_length:]
        
        # Reshape to (1, sequence_length, n_features)
        X = feature_data.reshape(1, self.sequence_length, -1)
        
        # Normalize
        X_normalized = self.normalize_data(X, fit=False)
        
        return X_normalized


def create_time_series_service(
    sequence_length: int = 24,
    forecast_horizons: List[int] = [24, 48, 72]
) -> TimeSeriesDataService:
    """
    Factory function to create TimeSeriesDataService.
    
    Args:
        sequence_length: Hours of historical data to use
        forecast_horizons: Forecast horizons in hours
        
    Returns:
        TimeSeriesDataService instance
    """
    return TimeSeriesDataService(sequence_length, forecast_horizons)
