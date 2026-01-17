# services/alert_formatter.py
"""
Alert Formatter Service - Phase 3

Converts Phase 2 predictions into CAP (Common Alerting Protocol) compliant alerts.
CAP is an international standard for emergency alerting that enables:
- Multi-language support
- Integration with emergency systems
- Automated distribution
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import uuid

logger = logging.getLogger("crisisconnect.alert_formatter")


class AlertUrgency(Enum):
    """Standard alert urgencies from common warning systems"""
    IMMEDIATE = "Immediate"   # Responsive action SHOULD be taken immediately
    EXPECTED = "Expected"     # Responsive action SHOULD be taken soon (within next hour)
    FUTURE = "Future"         # Responsive action SHOULD be taken in the near future
    PAST = "Past"             # Responsive action is no longer required
    UNKNOWN = "Unknown"       # Urgency not known


class AlertSeverity(Enum):
    """How severe is the hazard?"""
    EXTREME = "Extreme"       # Extraordinary threat to life or property
    SEVERE = "Severe"         # Significant threat to life or property
    MODERATE = "Moderate"     # Possible threat to life or property
    MINOR = "Minor"           # Minimal to no known threat
    UNKNOWN = "Unknown"       # Severity unknown


class AlertCertainty(Enum):
    """Certainty of the event occurring"""
    OBSERVED = "Observed"     # Determined to have occurred or to be ongoing
    LIKELY = "Likely"         # Likely (probability > ~50%)
    POSSIBLE = "Possible"     # Possible but not likely (probability <= ~50%)
    UNLIKELY = "Unlikely"     # Not expected to occur (probability ~ 0)
    UNKNOWN = "Unknown"       # Certainty unknown


class AlertStatus(Enum):
    """Status of the alert"""
    ACTUAL = "Actual"         # Actionable by all targeted recipients
    EXERCISE = "Exercise"     # Actionable only by designated exercise participants
    SYSTEM = "System"         # For messages that support alert network functions
    TEST = "Test"             # Technical testing only, all recipients disregard
    DRAFT = "Draft"           # A preliminary template or draft


class AlertMessageType(Enum):
    """Type of alert message"""
    ALERT = "Alert"           # Initial information requiring attention
    UPDATE = "Update"         # Updates and supersedes earlier message
    CANCEL = "Cancel"         # Cancels earlier message
    ACK = "Ack"               # Acknowledges receipt
    ERROR = "Error"           # Indicates rejection of earlier message


@dataclass
class AlertArea:
    """Geographic area affected by the alert"""
    area_desc: str                    # Text description of affected area
    polygon: Optional[List[Dict[str, float]]] = None  # Polygon coordinates
    circle: Optional[Dict] = None     # Circle definition (point + radius)
    geocode: Optional[Dict[str, str]] = None  # Standard geocode
    altitude: Optional[float] = None  # Altitude in meters
    ceiling: Optional[float] = None   # Maximum altitude
    
    # Extended fields for Crisis-Connect
    population: int = 0               # Estimated population in area
    evacuation_required: bool = False
    evacuation_zones: List[str] = field(default_factory=list)
    critical_facilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "areaDesc": self.area_desc,
            "polygon": self.polygon,
            "circle": self.circle,
            "geocode": self.geocode,
            "altitude": self.altitude,
            "ceiling": self.ceiling,
            "population": self.population,
            "evacuationRequired": self.evacuation_required,
            "evacuationZones": self.evacuation_zones,
            "criticalFacilities": self.critical_facilities
        }


@dataclass
class FormattedAlert:
    """
    CAP-compliant alert format
    
    Based on Common Alerting Protocol (CAP) v1.2
    Reference: http://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2.html
    """
    
    # Identification (required)
    identifier: str              # Unique ID for this alert
    sender: str                  # Identifier of the alert originator
    sent: str                    # Time alert was sent (ISO 8601)
    
    # Classification (required)
    status: AlertStatus          # Actual, Exercise, System, Test, Draft
    msg_type: AlertMessageType   # Alert, Update, Cancel, Ack, Error
    scope: str                   # Public, Restricted, Private
    
    # Optional identification
    source: Optional[str] = None      # Text identifying source
    restriction: Optional[str] = None # Restriction for restricted scope
    addresses: Optional[str] = None   # Addresses for private scope
    code: Optional[List[str]] = None  # Special handling codes
    note: Optional[str] = None        # Text description for non-standard use
    references: Optional[str] = None  # References to earlier alerts
    incidents: Optional[str] = None   # Incident identifiers
    
    # Info segment (the actual alert content)
    language: str = "en-US"
    category: str = "Met"             # Category: Geo, Met, Safety, Security, etc.
    event: str = ""                   # Type of event (e.g., "Flood Warning")
    response_type: str = "Prepare"    # Shelter, Evacuate, Prepare, Execute, etc.
    urgency: AlertUrgency = AlertUrgency.UNKNOWN
    severity: AlertSeverity = AlertSeverity.UNKNOWN
    certainty: AlertCertainty = AlertCertainty.UNKNOWN
    
    # Timing
    effective: Optional[str] = None   # Effective time
    onset: Optional[str] = None       # Expected onset time
    expires: Optional[str] = None     # Expiry time
    
    # Content
    sender_name: str = "Crisis-Connect System"
    headline: str = ""                # Brief headline
    description: str = ""             # Extended description
    instruction: str = ""             # Recommended action
    web: Optional[str] = None         # URL for more information
    contact: str = "Emergency Services: 10177"
    
    # Area
    areas: List[AlertArea] = field(default_factory=list)
    
    # Extended Crisis-Connect fields
    hazard_type: str = ""
    risk_score: float = 0.0
    hours_to_peak: int = 0
    method_breakdown: Dict[str, float] = field(default_factory=dict)
    progression_stage: str = ""
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to CAP-compliant dictionary"""
        return {
            "identifier": self.identifier,
            "sender": self.sender,
            "sent": self.sent,
            "status": self.status.value,
            "msgType": self.msg_type.value,
            "scope": self.scope,
            "source": self.source,
            "restriction": self.restriction,
            "addresses": self.addresses,
            "code": self.code,
            "note": self.note,
            "references": self.references,
            "incidents": self.incidents,
            "info": {
                "language": self.language,
                "category": self.category,
                "event": self.event,
                "responseType": self.response_type,
                "urgency": self.urgency.value,
                "severity": self.severity.value,
                "certainty": self.certainty.value,
                "effective": self.effective,
                "onset": self.onset,
                "expires": self.expires,
                "senderName": self.sender_name,
                "headline": self.headline,
                "description": self.description,
                "instruction": self.instruction,
                "web": self.web,
                "contact": self.contact,
                "area": [area.to_dict() for area in self.areas]
            },
            # Extended fields
            "crisisConnect": {
                "hazardType": self.hazard_type,
                "riskScore": self.risk_score,
                "hoursToPeak": self.hours_to_peak,
                "methodBreakdown": self.method_breakdown,
                "progressionStage": self.progression_stage,
                "confidence": self.confidence
            }
        }
    
    def to_simple_dict(self) -> Dict:
        """Convert to simplified dictionary for internal use"""
        return {
            "identifier": self.identifier,
            "sender": self.sender,
            "sent": self.sent,
            "status": self.status.value,
            "headline": self.headline,
            "description": self.description,
            "instruction": self.instruction,
            "severity": self.severity.value,
            "urgency": self.urgency.value,
            "hazard_type": self.hazard_type,
            "risk_score": self.risk_score,
            "hours_to_peak": self.hours_to_peak,
            "areas": [area.to_dict() for area in self.areas],
            "contact": self.contact
        }


class AlertFormatter:
    """
    Converts Phase 2 predictions into CAP-compliant alerts
    
    Uses CAP (Common Alerting Protocol) format which is:
    - International standard (OASIS)
    - Multi-language capable
    - Integrates with emergency systems worldwide
    - Automated distribution ready
    """
    
    def __init__(self, 
                 sender_id: str = "crisis-connect@emergency.gov.za",
                 default_web_url: str = "https://crisis-connect.gov.za"):
        self.sender_id = sender_id
        self.default_web_url = default_web_url
        self.alert_counter = 0
        
        # Hazard type to CAP category mapping
        self.hazard_categories = {
            "flood": "Met",
            "drought": "Met",
            "storm": "Met",
            "earthquake": "Geo",
            "wildfire": "Fire",
            "landslide": "Geo",
            "tsunami": "Geo"
        }
        
        # Response type mapping by risk level
        self.response_types = {
            "LOW": "Monitor",
            "MODERATE": "Prepare",
            "HIGH": "Prepare",
            "CRITICAL": "Evacuate"
        }
        
        logger.info(f"AlertFormatter initialized: sender={sender_id}")
    
    def format_prediction(self,
                         prediction: Dict,
                         location_data: Dict) -> FormattedAlert:
        """
        Convert Phase 2 prediction into CAP-compliant formatted alert
        
        Args:
            prediction: Output from Phase 2 AgentOrchestrator.predict()
                Expected keys: risk_score, risk_level, hours_to_peak, hazard_type,
                              location, prediction, forecasts, etc.
            location_data: Geographic and demographic info
                Expected keys: name, latitude, longitude, population, 
                              evacuation_zones, region, etc.
        
        Returns:
            FormattedAlert following CAP standard
        """
        # Extract prediction components
        pred_data = prediction.get('prediction', {})
        risk_score = pred_data.get('risk_score', prediction.get('risk_score', 0.0))
        risk_level = pred_data.get('risk_level', prediction.get('risk_level', 'LOW'))
        hours_to_peak = pred_data.get('hours_to_peak', prediction.get('hours_to_peak', 48))
        hazard_type = prediction.get('hazard_type', 'flood')
        location_id = prediction.get('location', location_data.get('name', 'Unknown'))
        
        # Get method breakdown if available
        method_breakdown = pred_data.get('method_breakdown', {})
        confidence = pred_data.get('confidence', 0.5)
        progression_stage = pred_data.get('progression_stage', 'unknown')
        
        # Determine alert parameters
        severity = self._score_to_severity(risk_score)
        urgency = self._hours_to_urgency(hours_to_peak)
        certainty = self._confidence_to_certainty(confidence)
        
        # Generate unique identifier
        self.alert_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        alert_id = f"CrisisConnect-{location_id.replace(' ', '_')}-{timestamp}-{self.alert_counter}"
        
        # Generate content
        headline = self._generate_headline(severity, hazard_type, location_data)
        description = self._generate_description(prediction, location_data)
        instruction = self._generate_instructions(risk_level, hazard_type, location_data, hours_to_peak)
        
        # Create affected areas
        areas = self._create_alert_areas(location_id, location_data, hazard_type, risk_level)
        
        # Calculate timing
        now = datetime.utcnow()
        effective = now.isoformat() + 'Z'
        onset = (now + timedelta(hours=max(0, hours_to_peak - 2))).isoformat() + 'Z'
        expires = (now + timedelta(hours=hours_to_peak + 24)).isoformat() + 'Z'
        
        # Create the alert
        alert = FormattedAlert(
            identifier=alert_id,
            sender=self.sender_id,
            sent=effective,
            status=AlertStatus.ACTUAL,
            msg_type=AlertMessageType.ALERT,
            scope="Public",
            source="Crisis-Connect AI Prediction System",
            category=self.hazard_categories.get(hazard_type.lower(), "Met"),
            event=f"{hazard_type.title()} Warning",
            response_type=self.response_types.get(risk_level, "Prepare"),
            urgency=urgency,
            severity=severity,
            certainty=certainty,
            effective=effective,
            onset=onset,
            expires=expires,
            headline=headline,
            description=description,
            instruction=instruction,
            web=f"{self.default_web_url}/alerts/{alert_id}",
            contact=self._get_emergency_contact(location_data),
            areas=areas,
            hazard_type=hazard_type,
            risk_score=risk_score,
            hours_to_peak=hours_to_peak,
            method_breakdown=method_breakdown,
            progression_stage=progression_stage,
            confidence=confidence
        )
        
        logger.info(f"Alert formatted: {alert_id}, severity={severity.value}, urgency={urgency.value}")
        
        return alert
    
    def _score_to_severity(self, risk_score: float) -> AlertSeverity:
        """Map risk score (0-1) to CAP alert severity"""
        if risk_score >= 0.75:
            return AlertSeverity.EXTREME
        elif risk_score >= 0.55:
            return AlertSeverity.SEVERE
        elif risk_score >= 0.30:
            return AlertSeverity.MODERATE
        elif risk_score > 0:
            return AlertSeverity.MINOR
        else:
            return AlertSeverity.UNKNOWN
    
    def _hours_to_urgency(self, hours: int) -> AlertUrgency:
        """Map hours to peak into CAP alert urgency"""
        if hours <= 6:
            return AlertUrgency.IMMEDIATE
        elif hours <= 24:
            return AlertUrgency.EXPECTED
        elif hours <= 72:
            return AlertUrgency.FUTURE
        else:
            return AlertUrgency.UNKNOWN
    
    def _confidence_to_certainty(self, confidence: float) -> AlertCertainty:
        """Map confidence score to CAP certainty"""
        if confidence >= 0.85:
            return AlertCertainty.OBSERVED
        elif confidence >= 0.65:
            return AlertCertainty.LIKELY
        elif confidence >= 0.40:
            return AlertCertainty.POSSIBLE
        else:
            return AlertCertainty.UNLIKELY
    
    def _generate_headline(self, 
                          severity: AlertSeverity,
                          hazard_type: str,
                          location_data: Dict) -> str:
        """Generate concise alert headline"""
        location_name = location_data.get('name', 'Unknown Location')
        region = location_data.get('region', '')
        
        if region and region != location_name:
            location_str = f"{location_name}, {region}"
        else:
            location_str = location_name
        
        return f"{severity.value} {hazard_type.title()} Warning for {location_str}"
    
    def _generate_description(self, prediction: Dict, location_data: Dict) -> str:
        """Generate detailed alert description with prediction reasoning"""
        pred_data = prediction.get('prediction', {})
        hazard = prediction.get('hazard_type', 'flood')
        risk_level = pred_data.get('risk_level', prediction.get('risk_level', 'LOW'))
        risk_score = pred_data.get('risk_score', prediction.get('risk_score', 0.0))
        hours = pred_data.get('hours_to_peak', prediction.get('hours_to_peak', 48))
        confidence = pred_data.get('confidence', 0.5)
        
        # Get method breakdown
        method_breakdown = pred_data.get('method_breakdown', {})
        
        # Get warnings
        warnings = pred_data.get('warnings', [])
        
        # Get recommendation from ensemble
        recommendation = pred_data.get('recommendation', '')
        
        description = f"""HAZARD: {hazard.upper()} WARNING

LOCATION: {location_data.get('name', 'Unknown')}
Region: {location_data.get('region', 'N/A')}
Population at Risk: {location_data.get('population', 'Unknown'):,}

RISK ASSESSMENT:
Risk Level: {risk_level}
Risk Score: {risk_score:.0%}
Estimated Peak: {hours} hours from now
Prediction Confidence: {confidence:.0%}

FORECAST BASIS:
This alert is generated by Crisis-Connect AI system combining:"""
        
        if method_breakdown:
            for method, score in method_breakdown.items():
                description += f"\n- {method.replace('_', ' ').title()}: {score:.0%} risk"
        else:
            description += "\n- Multi-agent ensemble analysis"
        
        if warnings:
            description += "\n\nWARNINGS:"
            for warning in warnings:
                description += f"\n- {warning}"
        
        if recommendation:
            description += f"\n\nSYSTEM RECOMMENDATION:\n{recommendation}"
        
        return description.strip()
    
    def _generate_instructions(self,
                              risk_level: str,
                              hazard_type: str,
                              location_data: Dict,
                              hours_to_peak: int) -> str:
        """Generate actionable instructions based on risk level"""
        
        if risk_level == "LOW":
            return """MONITOR CONDITIONS:
- Stay informed through official weather updates
- No immediate action required
- Be aware of changing conditions
- Review emergency plans as a precaution"""
        
        elif risk_level == "MODERATE":
            return f"""PREPARE NOW:
- Assemble emergency kit (water, food, first aid, flashlight, batteries)
- Charge mobile phones and power banks
- Keep important documents in waterproof container
- Know your evacuation routes
- Monitor official updates closely
- Secure outdoor items that could become projectiles

EXPECTED TIMELINE:
Peak {hazard_type} conditions expected in approximately {hours_to_peak} hours.
Be prepared to act on short notice if conditions worsen."""
        
        elif risk_level == "HIGH":
            evac_hours = max(1, hours_to_peak - 6)
            complete_hours = max(2, hours_to_peak - 2)
            
            return f"""PREPARE FOR EVACUATION:
- Pack essential belongings (medications, documents, valuables)
- Ensure vehicle is fueled and ready
- Know your designated evacuation zone and routes
- Identify nearest emergency shelter locations
- Arrange transportation for elderly and disabled family members
- Secure your property (turn off utilities if instructed)

EXPECTED TIMELINE:
- Evacuation orders may be issued in {evac_hours} hours
- Complete evacuation target: {complete_hours} hours before peak
- Peak conditions expected: {hours_to_peak} hours from now

FOLLOW ALL LOCAL EMERGENCY ORDERS IMMEDIATELY.
Do not wait if you feel unsafe - leave early if possible."""
        
        else:  # CRITICAL
            return f"""⚠️ EVACUATE IMMEDIATELY ⚠️

CONDITIONS ARE CRITICAL. Take action NOW.

IMMEDIATE ACTIONS:
1. Leave your location NOW for designated emergency shelter
2. Do NOT wait for official evacuation orders
3. Take only essential items (medications, ID, phone)
4. Follow marked evacuation routes
5. Avoid flooded roads - Turn Around Don't Drown
6. Help neighbors, especially elderly and disabled
7. Do NOT return until authorities declare it safe

PEAK CONDITIONS EXPECTED IN {hours_to_peak} HOURS
Areas may become inaccessible before peak.

EMERGENCY CONTACTS:
- Emergency Services: 10177
- Disaster Management: 0800 111 990
- Medical Emergency: 10177

IF TRAPPED:
- Move to highest point in building
- Signal for help (flashlight, whistle, bright cloth)
- Call emergency services with your location"""
    
    def _create_alert_areas(self,
                           location_id: str,
                           location_data: Dict,
                           hazard_type: str,
                           risk_level: str) -> List[AlertArea]:
        """Create affected area definitions"""
        areas = []
        
        # Main affected area
        main_area = AlertArea(
            area_desc=location_data.get('name', location_id),
            polygon=[{
                "latitude": location_data.get('latitude', 0),
                "longitude": location_data.get('longitude', 0)
            }] if location_data.get('latitude') else None,
            population=location_data.get('population', 0),
            evacuation_required=risk_level in ["HIGH", "CRITICAL"],
            evacuation_zones=location_data.get('evacuation_zones', []),
            critical_facilities=location_data.get('critical_facilities', [])
        )
        areas.append(main_area)
        
        # Add specific evacuation zones if available
        if 'evacuation_zones' in location_data:
            for zone in location_data.get('evacuation_zones', []):
                if isinstance(zone, dict):
                    zone_area = AlertArea(
                        area_desc=zone.get('name', 'Evacuation Zone'),
                        polygon=zone.get('polygon'),
                        population=zone.get('population', 0),
                        evacuation_required=True
                    )
                    areas.append(zone_area)
        
        return areas
    
    def _get_emergency_contact(self, location_data: Dict) -> str:
        """Get appropriate emergency contact for location"""
        region = location_data.get('region', '')
        
        # Default South African emergency contacts
        contacts = [
            "Emergency Services: 10177",
            "Disaster Management: 0800 111 990"
        ]
        
        # Add region-specific contacts if available
        if location_data.get('emergency_contact'):
            contacts.insert(0, location_data['emergency_contact'])
        
        return " | ".join(contacts)
    
    def create_update_alert(self, 
                           original_alert: FormattedAlert,
                           new_prediction: Dict,
                           location_data: Dict) -> FormattedAlert:
        """Create an update to an existing alert"""
        
        # Format new alert with updated prediction
        updated_alert = self.format_prediction(new_prediction, location_data)
        
        # Update metadata for update message
        updated_alert.msg_type = AlertMessageType.UPDATE
        updated_alert.references = f"{original_alert.sender},{original_alert.identifier},{original_alert.sent}"
        updated_alert.note = f"This message updates {original_alert.identifier}"
        
        logger.info(f"Update alert created for {original_alert.identifier}")
        
        return updated_alert
    
    def create_cancel_alert(self,
                           original_alert: FormattedAlert,
                           reason: str = "Conditions have improved") -> FormattedAlert:
        """Create a cancellation for an existing alert"""
        
        now = datetime.utcnow()
        self.alert_counter += 1
        
        cancel_alert = FormattedAlert(
            identifier=f"{original_alert.identifier}-CANCEL",
            sender=self.sender_id,
            sent=now.isoformat() + 'Z',
            status=AlertStatus.ACTUAL,
            msg_type=AlertMessageType.CANCEL,
            scope="Public",
            references=f"{original_alert.sender},{original_alert.identifier},{original_alert.sent}",
            headline=f"CANCELLED: {original_alert.headline}",
            description=f"The following alert has been cancelled:\n\n{original_alert.headline}\n\nReason: {reason}",
            instruction="The previous alert is no longer in effect. Resume normal activities but remain vigilant.",
            urgency=AlertUrgency.PAST,
            severity=AlertSeverity.MINOR,
            certainty=AlertCertainty.OBSERVED,
            areas=original_alert.areas,
            hazard_type=original_alert.hazard_type
        )
        
        logger.info(f"Cancel alert created for {original_alert.identifier}")
        
        return cancel_alert


# Factory function
def create_alert_formatter(sender_id: str = None, web_url: str = None) -> AlertFormatter:
    """Create an AlertFormatter instance with optional configuration"""
    return AlertFormatter(
        sender_id=sender_id or "crisis-connect@emergency.gov.za",
        default_web_url=web_url or "https://crisis-connect.gov.za"
    )
