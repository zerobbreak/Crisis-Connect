# services/alert_distributor.py
"""
Alert Distributor Service - Phase 3

Multi-channel alert distribution system that sends alerts through:
- SMS
- Email
- Dashboard/API
- Phone Tree (IVR)
- WhatsApp
- Emergency Sirens

Handles concurrent distribution and delivery tracking.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
import json

logger = logging.getLogger("crisisconnect.alert_distributor")


@dataclass
class DeliveryResult:
    """Result of a single delivery attempt"""
    channel: str
    recipient: str
    success: bool
    timestamp: datetime
    message_id: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "channel": self.channel,
            "recipient": self.recipient,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "error": self.error
        }


@dataclass
class DistributionReport:
    """Report of alert distribution results"""
    alert_id: str
    distribution_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    total_recipients: int
    successful_sends: int
    failed_sends: int
    results_by_channel: Dict[str, Dict]
    delivery_results: List[DeliveryResult]
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "distribution_id": self.distribution_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (self.completed_at - self.started_at).total_seconds() if self.completed_at else None,
            "total_recipients": self.total_recipients,
            "successful_sends": self.successful_sends,
            "failed_sends": self.failed_sends,
            "success_rate": self.successful_sends / self.total_recipients if self.total_recipients > 0 else 0,
            "results_by_channel": self.results_by_channel,
            "delivery_details": [r.to_dict() for r in self.delivery_results[:100]]  # Limit details
        }


class AlertChannel(ABC):
    """Base class for alert distribution channels"""
    
    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Name of this channel"""
        pass
    
    @abstractmethod
    async def send(self, alert: Dict, recipient: str) -> DeliveryResult:
        """
        Send alert to a single recipient
        
        Args:
            alert: Alert data to send
            recipient: Recipient identifier (phone, email, etc.)
            
        Returns:
            DeliveryResult with success/failure status
        """
        pass
    
    @abstractmethod
    def format_message(self, alert: Dict) -> str:
        """Format alert for this channel"""
        pass


class SMSChannel(AlertChannel):
    """Send alerts via SMS"""
    
    def __init__(self, api_key: str = None, sender_id: str = "CRISIS-ALERT"):
        self.api_key = api_key
        self.sender_id = sender_id
    
    @property
    def channel_name(self) -> str:
        return "sms"
    
    def format_message(self, alert: Dict) -> str:
        """Format alert for SMS (160 char limit for single SMS)"""
        headline = alert.get('headline', 'Emergency Alert')
        severity = alert.get('severity', 'Unknown')
        hours = alert.get('hours_to_peak', 'Unknown')
        
        # Truncate headline if needed
        max_headline = 60
        if len(headline) > max_headline:
            headline = headline[:max_headline-3] + "..."
        
        message = f"""⚠️ {headline}

Risk: {severity}
Peak: {hours}h

{alert.get('instruction', '')[:100]}...

Emergency: 10177
Reply HELP for info"""
        
        return message[:320]  # 2 SMS max
    
    async def send(self, alert: Dict, phone_number: str) -> DeliveryResult:
        """Send SMS alert"""
        try:
            message = self.format_message(alert)
            
            # In production: Use SMS provider API (Twilio, Africa's Talking, etc.)
            # For now, log the message
            logger.info(f"SMS to {phone_number}: {message[:50]}...")
            
            # Simulate API call
            await asyncio.sleep(0.1)
            
            return DeliveryResult(
                channel=self.channel_name,
                recipient=phone_number,
                success=True,
                timestamp=datetime.utcnow(),
                message_id=f"sms-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{phone_number[-4:]}"
            )
            
        except Exception as e:
            logger.error(f"SMS send failed to {phone_number}: {e}")
            return DeliveryResult(
                channel=self.channel_name,
                recipient=phone_number,
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e)
            )


class EmailChannel(AlertChannel):
    """Send alerts via email"""
    
    def __init__(self, smtp_config: Dict = None):
        self.smtp_config = smtp_config or {}
    
    @property
    def channel_name(self) -> str:
        return "email"
    
    def format_message(self, alert: Dict) -> str:
        """Format alert as HTML email"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #d32f2f; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .severity {{ font-size: 24px; font-weight: bold; }}
        .instruction {{ background-color: #fff3e0; padding: 15px; margin: 15px 0; }}
        .footer {{ font-size: 12px; color: #666; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚠️ EMERGENCY ALERT</h1>
        <p class="severity">{alert.get('headline', 'Emergency Alert')}</p>
    </div>
    <div class="content">
        <p><strong>Severity:</strong> {alert.get('severity', 'Unknown')}</p>
        <p><strong>Expected Peak:</strong> {alert.get('hours_to_peak', 'Unknown')} hours</p>
        
        <h2>Description</h2>
        <p>{alert.get('description', 'No description available')}</p>
        
        <div class="instruction">
            <h2>What You Should Do</h2>
            <p>{alert.get('instruction', 'Follow local emergency guidance')}</p>
        </div>
        
        <h2>Contact Information</h2>
        <p>{alert.get('contact', 'Emergency Services: 10177')}</p>
    </div>
    <div class="footer">
        <p>This alert was generated by Crisis-Connect Emergency Alert System</p>
        <p>Alert ID: {alert.get('identifier', 'Unknown')}</p>
        <p>Sent: {alert.get('sent', datetime.utcnow().isoformat())}</p>
    </div>
</body>
</html>
"""
    
    async def send(self, alert: Dict, email: str) -> DeliveryResult:
        """Send email alert"""
        try:
            html_content = self.format_message(alert)
            subject = f"⚠️ EMERGENCY: {alert.get('headline', 'Alert')}"
            
            # In production: Use email service (SMTP, SendGrid, etc.)
            logger.info(f"Email to {email}: {subject}")
            
            # Simulate API call
            await asyncio.sleep(0.1)
            
            return DeliveryResult(
                channel=self.channel_name,
                recipient=email,
                success=True,
                timestamp=datetime.utcnow(),
                message_id=f"email-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{hash(email) % 10000}"
            )
            
        except Exception as e:
            logger.error(f"Email send failed to {email}: {e}")
            return DeliveryResult(
                channel=self.channel_name,
                recipient=email,
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e)
            )


class DashboardChannel(AlertChannel):
    """Post alerts to emergency dashboard via API"""
    
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url or "http://localhost:8000/api/v1/dashboard"
        self.api_key = api_key
    
    @property
    def channel_name(self) -> str:
        return "dashboard"
    
    def format_message(self, alert: Dict) -> str:
        """Format as JSON for API"""
        return json.dumps(alert, default=str)
    
    async def send(self, alert: Dict, dashboard_id: str) -> DeliveryResult:
        """Post to dashboard API"""
        try:
            # In production: POST to dashboard API
            logger.info(f"Dashboard post to {dashboard_id}: {alert.get('headline', 'Alert')}")
            
            # Simulate API call
            await asyncio.sleep(0.05)
            
            return DeliveryResult(
                channel=self.channel_name,
                recipient=dashboard_id,
                success=True,
                timestamp=datetime.utcnow(),
                message_id=f"dash-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            )
            
        except Exception as e:
            logger.error(f"Dashboard post failed: {e}")
            return DeliveryResult(
                channel=self.channel_name,
                recipient=dashboard_id,
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e)
            )


class PhoneTreeChannel(AlertChannel):
    """Automated phone calls via IVR system"""
    
    def __init__(self, ivr_config: Dict = None):
        self.ivr_config = ivr_config or {}
    
    @property
    def channel_name(self) -> str:
        return "phone_tree"
    
    def format_message(self, alert: Dict) -> str:
        """Format for text-to-speech"""
        return f"""
This is an emergency alert from Crisis Connect.

{alert.get('headline', 'Emergency Alert')}.

Risk level is {alert.get('severity', 'elevated')}.

Peak conditions expected in {alert.get('hours_to_peak', 'unknown')} hours.

{alert.get('instruction', 'Please follow local emergency guidance')[:200]}

For more information, call emergency services at 10177.

Press 1 to repeat this message.
Press 2 to confirm receipt.
"""
    
    async def send(self, alert: Dict, phone_number: str) -> DeliveryResult:
        """Initiate automated call"""
        try:
            script = self.format_message(alert)
            
            # In production: Use IVR service (Twilio, etc.)
            logger.info(f"Phone call initiated to {phone_number}")
            
            # Simulate call initiation
            await asyncio.sleep(0.2)
            
            return DeliveryResult(
                channel=self.channel_name,
                recipient=phone_number,
                success=True,
                timestamp=datetime.utcnow(),
                message_id=f"call-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{phone_number[-4:]}"
            )
            
        except Exception as e:
            logger.error(f"Phone call failed to {phone_number}: {e}")
            return DeliveryResult(
                channel=self.channel_name,
                recipient=phone_number,
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e)
            )


class WhatsAppChannel(AlertChannel):
    """Send alerts via WhatsApp Business API"""
    
    def __init__(self, api_config: Dict = None):
        self.api_config = api_config or {}
    
    @property
    def channel_name(self) -> str:
        return "whatsapp"
    
    def format_message(self, alert: Dict) -> str:
        """Format for WhatsApp"""
        return f"""🚨 *EMERGENCY ALERT* 🚨

*{alert.get('headline', 'Emergency Alert')}*

📊 *Risk Level:* {alert.get('severity', 'Unknown')}
⏰ *Peak Expected:* {alert.get('hours_to_peak', 'Unknown')} hours

📋 *What to do:*
{alert.get('instruction', 'Follow local emergency guidance')[:500]}

📞 *Emergency:* 10177
🔗 *More info:* {alert.get('web', 'crisis-connect.gov.za')}

_Alert ID: {alert.get('identifier', 'Unknown')}_
"""
    
    async def send(self, alert: Dict, phone_number: str) -> DeliveryResult:
        """Send WhatsApp message"""
        try:
            message = self.format_message(alert)
            
            # In production: Use WhatsApp Business API
            logger.info(f"WhatsApp to {phone_number}: {message[:50]}...")
            
            # Simulate API call
            await asyncio.sleep(0.1)
            
            return DeliveryResult(
                channel=self.channel_name,
                recipient=phone_number,
                success=True,
                timestamp=datetime.utcnow(),
                message_id=f"wa-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{phone_number[-4:]}"
            )
            
        except Exception as e:
            logger.error(f"WhatsApp send failed to {phone_number}: {e}")
            return DeliveryResult(
                channel=self.channel_name,
                recipient=phone_number,
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e)
            )


class SirenChannel(AlertChannel):
    """Activate emergency sirens"""
    
    def __init__(self, siren_controller_url: str = None):
        self.controller_url = siren_controller_url
    
    @property
    def channel_name(self) -> str:
        return "siren"
    
    def format_message(self, alert: Dict) -> str:
        """Siren pattern based on severity"""
        severity = alert.get('severity', 'Unknown')
        patterns = {
            "Extreme": "CONTINUOUS",
            "Severe": "WAILING",
            "Moderate": "STEADY",
            "Minor": "NONE"
        }
        return patterns.get(severity, "STEADY")
    
    async def send(self, alert: Dict, siren_zone: str) -> DeliveryResult:
        """Activate sirens in zone"""
        try:
            pattern = self.format_message(alert)
            
            if pattern == "NONE":
                logger.info(f"Siren activation skipped for zone {siren_zone} - severity too low")
                return DeliveryResult(
                    channel=self.channel_name,
                    recipient=siren_zone,
                    success=True,
                    timestamp=datetime.utcnow(),
                    message_id=None
                )
            
            # In production: Connect to siren control system
            logger.warning(f"🚨 SIREN ACTIVATED in zone {siren_zone}: pattern={pattern}")
            
            # Simulate activation
            await asyncio.sleep(0.05)
            
            return DeliveryResult(
                channel=self.channel_name,
                recipient=siren_zone,
                success=True,
                timestamp=datetime.utcnow(),
                message_id=f"siren-{siren_zone}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            )
            
        except Exception as e:
            logger.error(f"Siren activation failed for zone {siren_zone}: {e}")
            return DeliveryResult(
                channel=self.channel_name,
                recipient=siren_zone,
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e)
            )


class TelegramChannel(AlertChannel):
    """Send alerts via Telegram (integrates with existing telegram_bot.py)"""
    
    def __init__(self, telegram_service=None):
        self.telegram_service = telegram_service
    
    @property
    def channel_name(self) -> str:
        return "telegram"
    
    def format_message(self, alert: Dict) -> str:
        """Format for Telegram"""
        return f"""🚨 *EMERGENCY ALERT* 🚨

*{alert.get('headline', 'Emergency Alert')}*

📊 Risk Level: {alert.get('severity', 'Unknown')}
⏰ Peak Expected: {alert.get('hours_to_peak', 'Unknown')} hours

📋 Instructions:
{alert.get('instruction', 'Follow local emergency guidance')[:500]}

📞 Emergency: 10177
"""
    
    async def send(self, alert: Dict, chat_id: str) -> DeliveryResult:
        """Send Telegram message"""
        try:
            message = self.format_message(alert)
            
            # Use existing telegram service if available
            if self.telegram_service:
                success = await self.telegram_service.send_alert(chat_id, message)
            else:
                # Log for testing
                logger.info(f"Telegram to {chat_id}: {message[:50]}...")
                success = True
            
            return DeliveryResult(
                channel=self.channel_name,
                recipient=chat_id,
                success=success,
                timestamp=datetime.utcnow(),
                message_id=f"tg-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{chat_id}"
            )
            
        except Exception as e:
            logger.error(f"Telegram send failed to {chat_id}: {e}")
            return DeliveryResult(
                channel=self.channel_name,
                recipient=chat_id,
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e)
            )


class AlertDistributor:
    """
    Distribute alerts through multiple channels simultaneously
    
    Ensures maximum reach by using multiple channels:
    - SMS for those without internet
    - Email for detailed information
    - WhatsApp for mobile users
    - Dashboard for authorities
    - Sirens for immediate local alert
    - Telegram for subscribers
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Initialize channels
        self.channels: Dict[str, AlertChannel] = {
            "sms": SMSChannel(api_key=config.get("sms_api_key")),
            "email": EmailChannel(smtp_config=config.get("smtp_config")),
            "dashboard": DashboardChannel(
                api_url=config.get("dashboard_api_url"),
                api_key=config.get("dashboard_api_key")
            ),
            "phone_tree": PhoneTreeChannel(ivr_config=config.get("ivr_config")),
            "whatsapp": WhatsAppChannel(api_config=config.get("whatsapp_config")),
            "siren": SirenChannel(siren_controller_url=config.get("siren_controller_url")),
            "telegram": TelegramChannel(telegram_service=config.get("telegram_service"))
        }
        
        # Channel preferences by recipient type
        self.channel_preferences = {
            "residential": ["sms", "whatsapp"],
            "authority": ["email", "dashboard", "phone_tree"],
            "critical_facility": ["email", "phone_tree", "sms"],
            "community_leader": ["whatsapp", "sms", "telegram"],
            "media": ["email", "dashboard"],
            "siren_zone": ["siren"]
        }
        
        self.distribution_counter = 0
        
        logger.info(f"AlertDistributor initialized with {len(self.channels)} channels")
    
    async def distribute_alert(self,
                               alert: Dict,
                               recipient_list: List[Dict]) -> DistributionReport:
        """
        Distribute alert to all recipients via appropriate channels
        
        Args:
            alert: Formatted alert dictionary
            recipient_list: List of recipient dicts with:
                - type: "residential", "authority", "critical_facility", etc.
                - identifier: phone, email, chat_id, etc.
                - channels: Optional list of specific channels to use
                - priority: "immediate", "urgent", "routine"
        
        Returns:
            DistributionReport with success/failure counts
        """
        self.distribution_counter += 1
        distribution_id = f"DIST-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self.distribution_counter}"
        
        started_at = datetime.utcnow()
        
        logger.info(f"Starting alert distribution {distribution_id} to {len(recipient_list)} recipients")
        
        # Group recipients by type
        by_type = self._group_by_type(recipient_list)
        
        # Create send tasks
        tasks = []
        
        for recipient_type, recipients in by_type.items():
            channels = self.channel_preferences.get(recipient_type, ["sms"])
            
            for recipient in recipients:
                # Use specific channels if provided, otherwise use defaults
                recipient_channels = recipient.get('channels', channels)
                
                for channel_name in recipient_channels:
                    if channel_name in self.channels:
                        tasks.append(
                            self._send_safe(alert, recipient, channel_name)
                        )
        
        # Send all messages concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        delivery_results = []
        for result in results:
            if isinstance(result, DeliveryResult):
                delivery_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Distribution task failed: {result}")
        
        # Calculate statistics
        successful = sum(1 for r in delivery_results if r.success)
        failed = len(delivery_results) - successful
        
        # Group by channel
        results_by_channel = {}
        for result in delivery_results:
            if result.channel not in results_by_channel:
                results_by_channel[result.channel] = {"sent": 0, "success": 0, "failed": 0}
            results_by_channel[result.channel]["sent"] += 1
            if result.success:
                results_by_channel[result.channel]["success"] += 1
            else:
                results_by_channel[result.channel]["failed"] += 1
        
        completed_at = datetime.utcnow()
        
        report = DistributionReport(
            alert_id=alert.get('identifier', 'unknown'),
            distribution_id=distribution_id,
            started_at=started_at,
            completed_at=completed_at,
            total_recipients=len(recipient_list),
            successful_sends=successful,
            failed_sends=failed,
            results_by_channel=results_by_channel,
            delivery_results=delivery_results
        )
        
        logger.info(
            f"Distribution {distribution_id} complete: "
            f"{successful} successful, {failed} failed "
            f"in {(completed_at - started_at).total_seconds():.2f}s"
        )
        
        return report
    
    async def _send_safe(self, 
                        alert: Dict, 
                        recipient: Dict, 
                        channel_name: str) -> DeliveryResult:
        """Send with error handling"""
        try:
            channel = self.channels.get(channel_name)
            if not channel:
                return DeliveryResult(
                    channel=channel_name,
                    recipient=recipient.get('identifier', 'unknown'),
                    success=False,
                    timestamp=datetime.utcnow(),
                    error=f"Channel {channel_name} not configured"
                )
            
            identifier = recipient.get('identifier')
            if not identifier:
                return DeliveryResult(
                    channel=channel_name,
                    recipient='unknown',
                    success=False,
                    timestamp=datetime.utcnow(),
                    error="No recipient identifier provided"
                )
            
            return await channel.send(alert, identifier)
            
        except Exception as e:
            logger.error(f"Send failed via {channel_name}: {e}")
            return DeliveryResult(
                channel=channel_name,
                recipient=recipient.get('identifier', 'unknown'),
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    def _group_by_type(self, recipients: List[Dict]) -> Dict[str, List[Dict]]:
        """Group recipients by type"""
        grouped = {}
        for recipient in recipients:
            type_key = recipient.get('type', 'residential')
            if type_key not in grouped:
                grouped[type_key] = []
            grouped[type_key].append(recipient)
        return grouped
    
    async def send_to_channel(self,
                             alert: Dict,
                             channel_name: str,
                             recipients: List[str]) -> List[DeliveryResult]:
        """Send alert to specific channel for multiple recipients"""
        if channel_name not in self.channels:
            logger.error(f"Channel {channel_name} not available")
            return []
        
        channel = self.channels[channel_name]
        tasks = [channel.send(alert, recipient) for recipient in recipients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if isinstance(r, DeliveryResult)]
    
    def add_channel(self, name: str, channel: AlertChannel):
        """Add a new distribution channel"""
        self.channels[name] = channel
        logger.info(f"Added channel: {name}")
    
    def remove_channel(self, name: str):
        """Remove a distribution channel"""
        if name in self.channels:
            del self.channels[name]
            logger.info(f"Removed channel: {name}")
    
    def get_available_channels(self) -> List[str]:
        """Get list of available channel names"""
        return list(self.channels.keys())


# Factory function
def create_alert_distributor(config: Dict = None) -> AlertDistributor:
    """Create an AlertDistributor instance"""
    return AlertDistributor(config=config)
