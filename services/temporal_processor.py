"""
Temporal Data Processor

Converts raw disaster database and weather data into labeled sequences
for training LSTM models and pattern detection agents.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger("crisisconnect.temporal_processor")


class TemporalDataProcessor:
    """
    Convert raw disaster database into sequences labeled with outcomes
    
    Challenge: We have isolated disaster events, not continuous time series
    Solution: Reconstruct what conditions were like before each disaster
    
    Creates:
    - Positive examples: Weather sequences that preceded disasters
    - Negative examples: Weather sequences during non-disaster periods
    """
    
    def __init__(self, 
                 lookback_days: int = 14,
                 disaster_window_days: int = 7):
        """
        Args:
            lookback_days: Days of weather data to include in each sequence
            disaster_window_days: Days after pattern to check for disaster
        """
        self.lookback_days = lookback_days
        self.disaster_window_days = disaster_window_days
        
        # Default feature columns for weather data
        self.weather_feature_columns = [
            'temp_c', 'temperature', 'humidity', 'wind_kph', 'wind_speed',
            'pressure_mb', 'pressure', 'precip_mm', 'rainfall', 'precipitation',
            'cloud', 'cloud_cover', 'soil_moisture'
        ]
        
        logger.info(f"TemporalDataProcessor initialized: lookback={lookback_days}d, window={disaster_window_days}d")
    
    def create_training_sequences(self,
                                  disaster_data: pd.DataFrame,
                                  weather_data: pd.DataFrame,
                                  balance_classes: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create sequences for LSTM training from disaster and weather data
        
        For each disaster:
        - Gather N days of weather BEFORE the disaster
        - Label as "1" (disaster coming)
        
        For non-disaster periods:
        - Find periods with no disasters
        - Label as "0" (no disaster)
        
        Args:
            disaster_data: DataFrame with disaster records (must have 'date' column)
            weather_data: DataFrame with daily weather data (must have 'date' column)
            balance_classes: Whether to balance positive/negative examples
            
        Returns:
            Tuple of (sequences_df, labels_array)
        """
        logger.info("Creating training sequences from disaster and weather data...")
        
        # Ensure date columns are datetime
        disaster_data = disaster_data.copy()
        weather_data = weather_data.copy()
        
        if 'date' in disaster_data.columns:
            disaster_data['date'] = pd.to_datetime(disaster_data['date'], errors='coerce')
        
        if 'date' in weather_data.columns:
            weather_data['date'] = pd.to_datetime(weather_data['date'], errors='coerce')
            weather_data = weather_data.sort_values('date')
        
        # Find available feature columns
        available_features = [col for col in self.weather_feature_columns if col in weather_data.columns]
        if not available_features:
            logger.warning("No standard weather features found, using all numeric columns")
            available_features = weather_data.select_dtypes(include=[np.number]).columns.tolist()
            available_features = [c for c in available_features if c not in ['date', 'year', 'month', 'day']]
        
        logger.info(f"Using {len(available_features)} weather features: {available_features[:5]}...")
        
        positive_sequences = []
        negative_sequences = []
        
        # Create positive examples (before disasters)
        positive_sequences = self._create_positive_sequences(
            disaster_data, weather_data, available_features
        )
        
        # Create negative examples (non-disaster periods)
        n_negative = len(positive_sequences) if balance_classes else len(positive_sequences) * 2
        negative_sequences = self._create_negative_sequences(
            disaster_data, weather_data, available_features, n_negative
        )
        
        # Combine sequences
        all_sequences = positive_sequences + negative_sequences
        labels = np.array([1] * len(positive_sequences) + [0] * len(negative_sequences))
        
        if not all_sequences:
            logger.warning("No sequences created")
            return pd.DataFrame(), np.array([])
        
        # Create DataFrame
        sequences_df = pd.concat(all_sequences, ignore_index=True)
        
        # Shuffle
        indices = np.random.permutation(len(labels))
        # Note: We can't easily shuffle the DataFrame rows to match labels
        # So we return them in order and let the caller handle shuffling
        
        logger.info(f"Created {len(positive_sequences)} positive and {len(negative_sequences)} negative sequences")
        logger.info(f"Total sequences: {len(all_sequences)}, Features: {len(available_features)}")
        
        return sequences_df, labels
    
    def _create_positive_sequences(self,
                                   disaster_data: pd.DataFrame,
                                   weather_data: pd.DataFrame,
                                   feature_columns: List[str]) -> List[pd.DataFrame]:
        """Create sequences that precede disasters (positive examples)"""
        sequences = []
        
        for idx, disaster in disaster_data.iterrows():
            if pd.isna(disaster.get('date')):
                continue
            
            disaster_date = pd.to_datetime(disaster['date'])
            
            # Get location identifier
            location = disaster.get('location') or disaster.get('country') or disaster.get('region')
            disaster_type = disaster.get('disaster_type', 'unknown')
            
            # Calculate window
            window_start = disaster_date - timedelta(days=self.lookback_days)
            window_end = disaster_date - timedelta(days=1)
            
            # Filter weather data for this location and time period
            if 'location' in weather_data.columns:
                location_weather = weather_data[weather_data['location'] == location]
            elif 'country' in weather_data.columns:
                location_weather = weather_data[weather_data['country'] == location]
            else:
                location_weather = weather_data
            
            # Filter by date
            window_data = location_weather[
                (location_weather['date'] >= window_start) &
                (location_weather['date'] <= window_end)
            ].copy()
            
            if len(window_data) >= self.lookback_days * 0.7:  # At least 70% of days
                # Select features
                sequence_features = window_data[feature_columns].copy()
                
                # Add metadata
                sequence_features['disaster_type'] = disaster_type
                sequence_features['disaster_date'] = disaster_date
                sequence_features['sequence_id'] = f"pos_{idx}"
                
                sequences.append(sequence_features)
        
        return sequences
    
    def _create_negative_sequences(self,
                                   disaster_data: pd.DataFrame,
                                   weather_data: pd.DataFrame,
                                   feature_columns: List[str],
                                   count: int) -> List[pd.DataFrame]:
        """Create sequences from non-disaster periods (negative examples)"""
        sequences = []
        
        # Get all disaster dates with buffer
        disaster_dates = pd.to_datetime(disaster_data['date'].dropna())
        disaster_windows = []
        for d in disaster_dates:
            window_start = d - timedelta(days=self.lookback_days + self.disaster_window_days)
            window_end = d + timedelta(days=self.disaster_window_days)
            disaster_windows.append((window_start, window_end))
        
        # Get date range of weather data
        min_date = weather_data['date'].min()
        max_date = weather_data['date'].max()
        
        attempts = 0
        max_attempts = count * 20
        
        while len(sequences) < count and attempts < max_attempts:
            attempts += 1
            
            # Random start date
            days_range = (max_date - min_date).days - self.lookback_days - 1
            if days_range <= 0:
                break
            
            random_offset = np.random.randint(0, days_range)
            start_date = min_date + timedelta(days=random_offset)
            end_date = start_date + timedelta(days=self.lookback_days)
            
            # Check if overlaps with any disaster window
            overlaps = False
            for d_start, d_end in disaster_windows:
                if not (end_date < d_start or start_date > d_end):
                    overlaps = True
                    break
            
            if overlaps:
                continue
            
            # Get weather data for this period
            window_data = weather_data[
                (weather_data['date'] >= start_date) &
                (weather_data['date'] <= end_date)
            ].copy()
            
            if len(window_data) >= self.lookback_days * 0.7:
                sequence_features = window_data[feature_columns].copy()
                sequence_features['disaster_type'] = 'none'
                sequence_features['disaster_date'] = pd.NaT
                sequence_features['sequence_id'] = f"neg_{len(sequences)}"
                
                sequences.append(sequence_features)
        
        return sequences
    
    def prepare_features(self, sequences_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features specifically for LSTM training
        
        LSTM works best with continuous, normalized features
        
        Args:
            sequences_df: DataFrame with weather sequences
            
        Returns:
            DataFrame with additional engineered features
        """
        df = sequences_df.copy()
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Skip metadata columns
        skip_cols = ['sequence_id', 'disaster_date', 'year', 'month', 'day']
        numeric_cols = [c for c in numeric_cols if c not in skip_cols]
        
        if not numeric_cols:
            return df
        
        # Add rolling features for key columns
        for col in numeric_cols[:5]:  # Limit to first 5 to avoid explosion
            if col in df.columns:
                # 3-day rolling mean
                df[f'{col}_roll3'] = df[col].rolling(3, min_periods=1).mean()
                
                # Rate of change
                df[f'{col}_change'] = df[col].diff().fillna(0)
        
        # Rainfall-specific features (if available)
        rain_col = next((c for c in numeric_cols if 'rain' in c.lower() or 'precip' in c.lower()), None)
        if rain_col:
            # Cumulative rainfall
            df['rainfall_cumsum'] = df[rain_col].cumsum()
            
            # Heavy rain indicator
            threshold = df[rain_col].quantile(0.75) if len(df) > 10 else 20
            df['heavy_rain_flag'] = (df[rain_col] > threshold).astype(int)
        
        # Fill NaN from rolling windows
        df = df.bfill().ffill().fillna(0)
        
        return df
    
    def create_sequences_from_master_dataset(self,
                                             master_df: pd.DataFrame,
                                             feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create LSTM-ready sequences from the Phase 1 master disaster dataset
        
        This is a simplified approach that uses the disaster records directly,
        treating each record as a snapshot that either did or didn't result in a disaster.
        
        Args:
            master_df: The disasters_master.csv DataFrame
            feature_columns: Columns to use as features (auto-detected if None)
            
        Returns:
            Tuple of (X_sequences, y_labels, feature_names)
        """
        logger.info("Creating sequences from master disaster dataset...")
        
        df = master_df.copy()
        
        # Auto-detect numeric feature columns
        if feature_columns is None:
            # Get numeric columns, excluding obvious non-features
            exclude_cols = [
                'disaster_id', 'event_name', 'country', 'location', 'region',
                'date', 'start_date', 'end_date', 'disaster_type', 'subtype',
                'year', 'month', 'day', 'sequence_id', 'disaster_date'
            ]
            feature_columns = [
                col for col in df.select_dtypes(include=[np.number]).columns
                if col.lower() not in [c.lower() for c in exclude_cols]
            ]
        
        logger.info(f"Using {len(feature_columns)} features: {feature_columns[:10]}...")
        
        # Ensure all feature columns exist
        feature_columns = [c for c in feature_columns if c in df.columns]
        
        if not feature_columns:
            logger.error("No valid feature columns found")
            return np.array([]), np.array([]), []
        
        # Extract features
        X = df[feature_columns].values
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create labels based on disaster occurrence
        # If there's a 'deaths' or 'affected' column, use it to determine severity
        if 'deaths' in df.columns:
            y = (df['deaths'] > 0).astype(int).values
        elif 'affected' in df.columns:
            y = (df['affected'] > 0).astype(int).values
        else:
            # All records are disasters
            y = np.ones(len(df), dtype=int)
        
        logger.info(f"Created {len(X)} samples with {X.shape[1]} features")
        logger.info(f"Label distribution: {np.sum(y == 1)} positive, {np.sum(y == 0)} negative")
        
        return X, y, feature_columns
    
    def get_sequence_stats(self, sequences_df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Get statistics about the created sequences"""
        stats = {
            "total_sequences": len(labels),
            "positive_sequences": int(np.sum(labels == 1)),
            "negative_sequences": int(np.sum(labels == 0)),
            "balance_ratio": float(np.sum(labels == 1) / len(labels)) if len(labels) > 0 else 0,
            "num_features": len(sequences_df.columns) if len(sequences_df) > 0 else 0,
            "feature_columns": list(sequences_df.columns) if len(sequences_df) > 0 else []
        }
        
        return stats
