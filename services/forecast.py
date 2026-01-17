"""
Forecast Service - Time Series Flood Risk Forecasting
Integrates LSTM model with existing infrastructure for multi-horizon predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

from services.time_series_data import TimeSeriesDataService
from services.lstm_model import LSTMForecastModel, load_pretrained_model

logger = logging.getLogger("crisisconnect.forecast")


class ForecastService:
    """
    Service for generating flood risk forecasts using LSTM model.
    Provides multi-horizon predictions with confidence intervals.
    """
    
    def __init__(
        self,
        model_path: str = "data/models/lstm_forecast.h5",
        sequence_length: int = 24,
        forecast_horizons: List[int] = [24, 48, 72]
    ):
        """
        Initialize forecast service.
        
        Args:
            model_path: Path to trained LSTM model
            sequence_length: Hours of historical data needed
            forecast_horizons: Forecast horizons in hours
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.forecast_horizons = forecast_horizons
        
        # Initialize data service
        self.data_service = TimeSeriesDataService(
            sequence_length=sequence_length,
            forecast_horizons=forecast_horizons
        )
        
        # Load model if available
        self.model = None
        self._load_model()
        
        logger.info(f"ForecastService initialized: horizons={forecast_horizons}")
    
    def _load_model(self):
        """Load pre-trained LSTM model."""
        if Path(self.model_path).exists():
            try:
                self.model = load_pretrained_model(self.model_path)
                logger.info(f"✅ Loaded LSTM model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}")
                self.model = None
        else:
            logger.warning(f"LSTM model not found at {self.model_path}")
            self.model = None
    
    def generate_forecast(
        self,
        location_id: str,
        location_name: str,
        lat: float,
        lon: float,
        recent_weather: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Generate multi-horizon forecast for a location.
        
        Args:
            location_id: Location identifier
            location_name: Location name
            lat: Latitude
            lon: Longitude
            recent_weather: Recent weather data (if None, will fetch)
            
        Returns:
            Forecast dictionary with predictions and metadata
        """
        if self.model is None:
            return self._fallback_forecast(location_id, location_name)
        
        try:
            # Get recent weather data if not provided
            if recent_weather is None:
                recent_weather = self._fetch_recent_weather(lat, lon, location_name)
            
            # Prepare data for inference
            X = self.data_service.prepare_inference_data(recent_weather)
            
            # Generate prediction
            predictions, confidence = self.model.predict_single(X)
            
            # Calculate current risk (from most recent data)
            current_risk = self._calculate_current_risk(recent_weather)
            
            # Build forecast response
            forecasts = []
            for i, horizon in enumerate(self.forecast_horizons):
                risk_score = float(predictions[i])
                
                # Calculate confidence intervals (±10% for now, can be improved)
                confidence_range = 10.0
                
                # Determine trend
                trend = self._determine_trend(current_risk, risk_score)
                
                # Generate explanation
                explanation = self._generate_explanation(risk_score, trend, horizon)
                
                forecasts.append({
                    "horizon_hours": horizon,
                    "predicted_risk": round(risk_score, 2),
                    "confidence_lower": round(max(0, risk_score - confidence_range), 2),
                    "confidence_upper": round(min(100, risk_score + confidence_range), 2),
                    "trend": trend,
                    "risk_category": self._categorize_risk(risk_score),
                    "explanation": explanation
                })
            
            # Detect early warnings
            early_warning = self._detect_early_warning(forecasts, current_risk)
            
            return {
                "location_id": location_id,
                "location_name": location_name,
                "current_risk": round(current_risk, 2),
                "forecasts": forecasts,
                "early_warning": early_warning,
                "generated_at": datetime.now().isoformat(),
                "model_type": "LSTM",
                "forecast_available": True
            }
            
        except Exception as e:
            logger.error(f"Forecast generation failed for {location_name}: {e}")
            return self._fallback_forecast(location_id, location_name)
    
    def get_forecast_timeline(
        self,
        location_id: str,
        location_name: str,
        lat: float,
        lon: float,
        hours: int = 72
    ) -> Dict:
        """
        Get hourly risk timeline for next N hours.
        
        Args:
            location_id: Location identifier
            location_name: Location name
            lat: Latitude
            lon: Longitude
            hours: Number of hours to forecast
            
        Returns:
            Timeline dictionary with hourly predictions
        """
        if self.model is None:
            return {"error": "LSTM model not available", "timeline": []}
        
        try:
            # Get recent weather
            recent_weather = self._fetch_recent_weather(lat, lon, location_name)
            
            # Generate forecasts
            forecast = self.generate_forecast(location_id, location_name, lat, lon, recent_weather)
            
            # Interpolate hourly values from horizon predictions
            timeline = self._interpolate_timeline(forecast, hours)
            
            return {
                "location_id": location_id,
                "location_name": location_name,
                "timeline": timeline,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Timeline generation failed: {e}")
            return {"error": str(e), "timeline": []}
    
    def _fetch_recent_weather(
        self,
        lat: float,
        lon: float,
        location_name: str
    ) -> pd.DataFrame:
        """Fetch recent weather data for inference."""
        # Fetch last 24 hours of data
        df = self.data_service.fetch_historical_weather(
            lat, lon,
            days_back=2,  # Get 2 days to ensure we have 24 hours
            location_name=location_name
        )
        
        # Return last 24 hours
        return df.tail(self.sequence_length)
    
    def _calculate_current_risk(self, recent_weather: pd.DataFrame) -> float:
        """Calculate current risk from recent weather data."""
        # Simple heuristic based on latest weather
        latest = recent_weather.iloc[-1]
        
        risk = 0.0
        
        # Precipitation factor
        if latest['precip_mm'] > 50:
            risk += 40
        elif latest['precip_mm'] > 20:
            risk += 25
        elif latest['precip_mm'] > 10:
            risk += 15
        
        # Wind factor
        if latest['wind_kph'] > 60:
            risk += 30
        elif latest['wind_kph'] > 40:
            risk += 20
        
        # Humidity factor
        if latest['humidity'] > 90:
            risk += 15
        elif latest['humidity'] > 80:
            risk += 10
        
        # Wave height (if coastal)
        if 'wave_height' in latest and latest['wave_height'] > 2:
            risk += 15
        
        return min(risk, 100)
    
    def _determine_trend(self, current_risk: float, future_risk: float) -> str:
        """Determine trend direction."""
        diff = future_risk - current_risk
        
        if diff > 10:
            return "increasing"
        elif diff < -10:
            return "decreasing"
        else:
            return "stable"
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score."""
        if risk_score >= 75:
            return "Very High"
        elif risk_score >= 60:
            return "High"
        elif risk_score >= 40:
            return "Medium"
        elif risk_score >= 25:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_explanation(self, risk_score: float, trend: str, horizon: int) -> str:
        """Generate human-readable explanation."""
        category = self._categorize_risk(risk_score)
        
        explanations = {
            "increasing": f"{category} risk expected in {horizon}h (conditions worsening)",
            "decreasing": f"{category} risk expected in {horizon}h (conditions improving)",
            "stable": f"{category} risk expected in {horizon}h (conditions stable)"
        }
        
        return explanations.get(trend, f"{category} risk expected in {horizon}h")
    
    def _detect_early_warning(self, forecasts: List[Dict], current_risk: float) -> bool:
        """Detect if early warning should be issued."""
        # Check if any forecast shows significant risk increase
        for forecast in forecasts:
            if forecast["predicted_risk"] > 70:  # High risk threshold
                return True
            if forecast["predicted_risk"] - current_risk > 20:  # Rapid increase
                return True
        
        return False
    
    def _interpolate_timeline(self, forecast: Dict, hours: int) -> List[Dict]:
        """Interpolate hourly timeline from horizon forecasts."""
        timeline = []
        current_time = datetime.now()
        current_risk = forecast["current_risk"]
        
        # Get forecast points
        forecast_points = {0: current_risk}
        for f in forecast["forecasts"]:
            forecast_points[f["horizon_hours"]] = f["predicted_risk"]
        
        # Interpolate hourly
        for hour in range(1, hours + 1):
            # Find surrounding forecast points
            lower_hour = max([h for h in forecast_points.keys() if h <= hour], default=0)
            upper_hour = min([h for h in forecast_points.keys() if h >= hour], default=max(forecast_points.keys()))
            
            # Linear interpolation
            if lower_hour == upper_hour:
                risk = forecast_points[lower_hour]
            else:
                ratio = (hour - lower_hour) / (upper_hour - lower_hour)
                risk = forecast_points[lower_hour] + ratio * (forecast_points[upper_hour] - forecast_points[lower_hour])
            
            timeline.append({
                "timestamp": (current_time + timedelta(hours=hour)).isoformat(),
                "hour_offset": hour,
                "risk_score": round(risk, 2),
                "risk_category": self._categorize_risk(risk)
            })
        
        return timeline
    
    def _fallback_forecast(self, location_id: str, location_name: str) -> Dict:
        """Provide fallback forecast when LSTM is unavailable."""
        return {
            "location_id": location_id,
            "location_name": location_name,
            "current_risk": 0.0,
            "forecasts": [],
            "early_warning": False,
            "generated_at": datetime.now().isoformat(),
            "model_type": "None",
            "forecast_available": False,
            "message": "LSTM model not available. Please train the model first."
        }
    
    def generate_batch_forecasts(
        self,
        locations: List[Dict[str, any]]
    ) -> List[Dict]:
        """
        Generate forecasts for multiple locations.
        
        Args:
            locations: List of location dicts with id, name, lat, lon
            
        Returns:
            List of forecast dictionaries
        """
        forecasts = []
        
        for location in locations:
            try:
                forecast = self.generate_forecast(
                    location_id=location.get("id", ""),
                    location_name=location.get("name", "Unknown"),
                    lat=location.get("lat", 0.0),
                    lon=location.get("lon", 0.0)
                )
                forecasts.append(forecast)
            except Exception as e:
                logger.error(f"Batch forecast failed for {location.get('name')}: {e}")
                continue
        
        return forecasts


def create_forecast_service(
    model_path: str = "data/models/lstm_forecast.h5",
    **kwargs
) -> ForecastService:
    """
    Factory function to create ForecastService.
    
    Args:
        model_path: Path to LSTM model
        **kwargs: Additional arguments
        
    Returns:
        ForecastService instance
    """
    return ForecastService(model_path=model_path, **kwargs)
