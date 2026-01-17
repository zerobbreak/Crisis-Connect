"""
Alert Engine Service
Evaluates risk and triggers alerts based on forecasts and real-time data.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

from services.telegram_bot import telegram_service
from services.flood_data import flood_service

logger = logging.getLogger("crisisconnect.alert_engine")

class AlertEngine:
    """
    Core logic for detecting high-risk scenarios and triggering alerts.
    """
    
    def __init__(self):
        # Thresholds
        self.RISK_THRESHOLD_HIGH = 70.0
        self.RISK_THRESHOLD_MEDIUM = 40.0
        self.RIVER_DISCHARGE_THRESHOLD_PERCENTILE = 90.0 # Top 10% flow is dangerous
        
        # Cache for deduplication (simple in-memory for now)
        self.sent_alerts: Dict[str, datetime] = {}
        self.dedup_window_seconds = 3600 * 6 # 6 hours

    async def evaluate_risk(self, location: Dict, forecast: Dict) -> Optional[Dict]:
        """
        Evaluate if a location needs an alert based on forecast.
        
        Args:
            location: Location document
            forecast: Forecast result from ForecastService
            
        Returns:
            Alert details dict if risk is high, None otherwise
        """
        location_name = location.get("name", "Unknown")
        lat = location.get("latitude")
        lon = location.get("longitude")
        
        # 1. Check Forecast Risk Scores
        max_risk = 0.0
        risk_horizon = 0
        
        if "forecasts" in forecast:
            for f in forecast["forecasts"]:
                if f["predicted_risk"] > max_risk:
                    max_risk = f["predicted_risk"]
                    risk_horizon = f["horizon_hours"]
        
        # 2. Check River Discharge (Ground Truth)
        river_risk = False
        river_details = ""
        
        try:
            flood_data = await flood_service.fetch_river_discharge(lat, lon, days_forecast=1)
            if flood_data and "daily" in flood_data:
                # Simple check: is current discharge high? 
                # In a real app, we'd compare to historical percentiles.
                # Here we just check if it exists and is non-zero as a proxy for "data available"
                # and assume if forecast is high + river is flowing, it's bad.
                current_discharge = flood_data["daily"]["river_discharge"][0]
                if current_discharge and current_discharge > 50.0: # Arbitrary threshold for demo
                    river_risk = True
                    river_details = f"(River Discharge: {current_discharge:.1f} m³/s)"
        except Exception as e:
            logger.warning(f"Failed to check river risk for {location_name}: {e}")

        # 3. Decision Logic
        should_alert = False
        severity = "LOW"
        message = ""
        
        if max_risk >= self.RISK_THRESHOLD_HIGH:
            should_alert = True
            severity = "HIGH"
            message = f"🚨 HIGH FLOOD RISK detected for {location_name}!\n" \
                      f"Risk Score: {max_risk:.1f}/100 in {risk_horizon} hours.\n" \
                      f"{river_details}\n" \
                      f"Take immediate precautions."
                      
        elif max_risk >= self.RISK_THRESHOLD_MEDIUM and river_risk:
            should_alert = True
            severity = "MEDIUM"
            message = f"⚠️ Elevated Flood Risk for {location_name}.\n" \
                      f"Risk Score: {max_risk:.1f}/100.\n" \
                      f"River levels are high {river_details}."

        if should_alert:
            return {
                "location_id": str(location["_id"]),
                "location_name": location_name,
                "severity": severity,
                "message": message,
                "timestamp": datetime.now()
            }
            
        return None

    async def process_alert(self, alert_data: Dict):
        """
        Process a generated alert: Deduplicate and Send.
        """
        alert_key = f"{alert_data['location_id']}_{alert_data['severity']}"
        
        # Deduplication check
        if alert_key in self.sent_alerts:
            last_sent = self.sent_alerts[alert_key]
            if (datetime.now() - last_sent).total_seconds() < self.dedup_window_seconds:
                logger.info(f"Skipping duplicate alert for {alert_data['location_name']}")
                return

        # Send Notification
        # For this demo we use a fixed chat ID; in production this would be user‑specific
        chat_id = "TEST_CHAT_ID"
        await telegram_service.send_alert(chat_id, alert_data["message"])
        
        # Update dedup cache
        self.sent_alerts[alert_key] = datetime.now()
        
        # TODO: Integrate with Email or Webhook notification channels

# Singleton instance
alert_engine = AlertEngine()
