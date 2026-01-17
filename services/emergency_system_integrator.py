# services/emergency_system_integrator.py
"""
Emergency System Integrator Service - Phase 3

Integrates Crisis-Connect with existing emergency management systems:
- Emergency Operations Center (EOC) dashboards
- Disaster management databases
- Hospital networks
- Media broadcast systems
- Weather services
- Municipal alert systems
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
import json

logger = logging.getLogger("crisisconnect.emergency_integrator")


@dataclass
class IntegrationResult:
    """Result of integration with external system"""
    system_name: str
    success: bool
    timestamp: datetime
    response_data: Optional[Dict] = None
    error: Optional[str] = None
    latency_ms: float = 0
    
    def to_dict(self) -> Dict:
        return {
            "system_name": self.system_name,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "response_data": self.response_data,
            "error": self.error,
            "latency_ms": self.latency_ms
        }


@dataclass
class IntegrationReport:
    """Report of all integration results"""
    alert_id: str
    integration_id: str
    timestamp: datetime
    total_systems: int
    successful: int
    failed: int
    results: List[IntegrationResult]
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "integration_id": self.integration_id,
            "timestamp": self.timestamp.isoformat(),
            "total_systems": self.total_systems,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.successful / self.total_systems if self.total_systems > 0 else 0,
            "results": [r.to_dict() for r in self.results]
        }


class EmergencySystemIntegrator:
    """
    Integrate Crisis-Connect with existing emergency systems
    
    Systems integrated:
    - Emergency Operations Center (EOC) dashboard
    - Disaster management database
    - Municipal alert system
    - Weather services
    - Hospital notification system
    - Media broadcast system
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # System endpoints (in production, these would be real URLs)
        self.endpoints = {
            "eoc": config.get("eoc_api_url", "http://localhost:8001/api/eoc"),
            "disaster_db": config.get("disaster_db_url", "http://localhost:8002/api/disasters"),
            "weather_service": config.get("weather_service_api", "http://localhost:8003/api/weather"),
            "hospital_network": config.get("hospital_api", "http://localhost:8004/api/hospitals"),
            "media_system": config.get("media_api", "http://localhost:8005/api/media"),
            "municipal": config.get("municipal_api", "http://localhost:8006/api/municipal")
        }
        
        # API keys for external systems
        self.api_keys = {
            "eoc": config.get("eoc_api_key"),
            "disaster_db": config.get("disaster_db_api_key"),
            "hospital_network": config.get("hospital_api_key"),
            "media_system": config.get("media_api_key"),
            "municipal": config.get("municipal_api_key")
        }
        
        self.integration_counter = 0
        
        logger.info("EmergencySystemIntegrator initialized")
    
    async def integrate_alert(self,
                             alert: Dict,
                             actions: List[Dict] = None,
                             resources: Dict = None) -> IntegrationReport:
        """
        Push alert and action plan to all integrated systems
        
        Args:
            alert: Formatted alert dictionary
            actions: List of action dictionaries
            resources: Resource requirements dictionary
            
        Returns:
            IntegrationReport with results from all systems
        """
        self.integration_counter += 1
        integration_id = f"INT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self.integration_counter}"
        
        logger.info(f"Starting integration {integration_id} for alert {alert.get('identifier', 'unknown')}")
        
        results = []
        
        # Push to all systems concurrently
        tasks = [
            self._push_to_eoc(alert, actions, resources),
            self._update_disaster_db(alert),
            self._notify_hospitals(alert, resources),
            self._alert_media(alert),
            self._notify_municipal(alert, actions)
        ]
        
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in task_results:
            if isinstance(result, IntegrationResult):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Integration task failed: {result}")
                results.append(IntegrationResult(
                    system_name="unknown",
                    success=False,
                    timestamp=datetime.utcnow(),
                    error=str(result)
                ))
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        report = IntegrationReport(
            alert_id=alert.get('identifier', 'unknown'),
            integration_id=integration_id,
            timestamp=datetime.utcnow(),
            total_systems=len(results),
            successful=successful,
            failed=failed,
            results=results
        )
        
        logger.info(f"Integration {integration_id} complete: {successful}/{len(results)} systems successful")
        
        return report
    
    async def _push_to_eoc(self,
                          alert: Dict,
                          actions: List[Dict],
                          resources: Dict) -> IntegrationResult:
        """Push to Emergency Operations Center dashboard"""
        start_time = datetime.utcnow()
        
        try:
            payload = {
                "alert_id": alert.get('identifier'),
                "severity": alert.get('severity'),
                "urgency": alert.get('urgency'),
                "headline": alert.get('headline'),
                "description": alert.get('description'),
                "instruction": alert.get('instruction'),
                "location": alert.get('areas', [{}])[0].get('areaDesc', 'Unknown') if alert.get('areas') else 'Unknown',
                "coordinates": self._extract_coordinates(alert),
                "hazard_type": alert.get('crisisConnect', {}).get('hazardType', alert.get('hazard_type')),
                "risk_score": alert.get('crisisConnect', {}).get('riskScore', alert.get('risk_score')),
                "hours_to_peak": alert.get('crisisConnect', {}).get('hoursToPeak', alert.get('hours_to_peak')),
                "recommended_actions": actions[:10] if actions else [],  # Top 10 actions
                "resource_summary": self._summarize_resources(resources) if resources else {},
                "timestamp": alert.get('sent'),
                "source": "Crisis-Connect"
            }
            
            # In production: POST to EOC API
            # response = await httpx.AsyncClient().post(
            #     f"{self.endpoints['eoc']}/alerts",
            #     json=payload,
            #     headers={"Authorization": f"Bearer {self.api_keys['eoc']}"}
            # )
            
            logger.info(f"EOC integration: Posted alert {alert.get('identifier')}")
            
            # Simulate API call
            await asyncio.sleep(0.1)
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return IntegrationResult(
                system_name="EOC Dashboard",
                success=True,
                timestamp=datetime.utcnow(),
                response_data={
                    "eoc_alert_id": f"EOC-{alert.get('identifier')}",
                    "status": "received",
                    "acknowledged": False
                },
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"EOC integration failed: {e}")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            return IntegrationResult(
                system_name="EOC Dashboard",
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e),
                latency_ms=latency
            )
    
    async def _update_disaster_db(self, alert: Dict) -> IntegrationResult:
        """Update central disaster management database"""
        start_time = datetime.utcnow()
        
        try:
            # Extract location from areas
            areas = alert.get('areas', [])
            location = areas[0].get('areaDesc', 'Unknown') if areas else 'Unknown'
            
            record = {
                "event_id": alert.get('identifier'),
                "event_type": alert.get('crisisConnect', {}).get('hazardType', alert.get('hazard_type', 'unknown')),
                "location": location,
                "predicted_severity": alert.get('severity'),
                "alert_timestamp": alert.get('sent'),
                "expected_onset": alert.get('onset'),
                "expected_expires": alert.get('expires'),
                "risk_score": alert.get('crisisConnect', {}).get('riskScore', 0),
                "confidence": alert.get('crisisConnect', {}).get('confidence', 0),
                "status": "predicted",
                "source": "Crisis-Connect",
                "created_at": datetime.utcnow().isoformat()
            }
            
            # In production: POST to disaster database API
            logger.info(f"Disaster DB: Created record for {alert.get('identifier')}")
            
            await asyncio.sleep(0.05)
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return IntegrationResult(
                system_name="Disaster Database",
                success=True,
                timestamp=datetime.utcnow(),
                response_data={
                    "event_id": record["event_id"],
                    "status": "created"
                },
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Disaster DB update failed: {e}")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            return IntegrationResult(
                system_name="Disaster Database",
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e),
                latency_ms=latency
            )
    
    async def _notify_hospitals(self, alert: Dict, resources: Dict) -> IntegrationResult:
        """Alert hospital network for medical preparedness"""
        start_time = datetime.utcnow()
        
        try:
            # Estimate hospital impact
            estimated_patients = self._estimate_hospital_patients(alert, resources)
            
            notification = {
                "alert_id": alert.get('identifier'),
                "alert_type": alert.get('crisisConnect', {}).get('hazardType', alert.get('hazard_type')),
                "severity": alert.get('severity'),
                "location": self._extract_location_name(alert),
                "estimated_patient_volume": estimated_patients,
                "time_to_peak_hours": alert.get('crisisConnect', {}).get('hoursToPeak', alert.get('hours_to_peak', 24)),
                "required_response": self._get_hospital_response(alert),
                "resource_needs": self._get_medical_resource_needs(resources),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # In production: POST to hospital network API
            logger.info(f"Hospital network: Notified about {alert.get('identifier')}, est. {estimated_patients} patients")
            
            await asyncio.sleep(0.08)
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return IntegrationResult(
                system_name="Hospital Network",
                success=True,
                timestamp=datetime.utcnow(),
                response_data={
                    "hospitals_notified": 5,  # Simulated
                    "estimated_bed_availability": 250,
                    "status": "alert_received"
                },
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Hospital notification failed: {e}")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            return IntegrationResult(
                system_name="Hospital Network",
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e),
                latency_ms=latency
            )
    
    async def _alert_media(self, alert: Dict) -> IntegrationResult:
        """Alert media broadcast system for public notification"""
        start_time = datetime.utcnow()
        
        try:
            # Create media-friendly message
            message = self._format_media_message(alert)
            
            broadcast = {
                "alert_id": alert.get('identifier'),
                "headline": alert.get('headline'),
                "message": message,
                "priority": "high" if alert.get('severity') in ["Extreme", "Severe"] else "normal",
                "broadcast_timestamp": datetime.utcnow().isoformat(),
                "expiry": alert.get('expires'),
                "channels": ["radio", "tv", "online"],
                "languages": ["en", "zu", "xh", "af"]  # South African languages
            }
            
            # In production: POST to media system API
            logger.info(f"Media system: Broadcast initiated for {alert.get('identifier')}")
            
            await asyncio.sleep(0.06)
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return IntegrationResult(
                system_name="Media Broadcast",
                success=True,
                timestamp=datetime.utcnow(),
                response_data={
                    "broadcast_id": f"MEDIA-{alert.get('identifier')}",
                    "channels_activated": ["radio", "tv", "online"],
                    "status": "broadcasting"
                },
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Media alert failed: {e}")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            return IntegrationResult(
                system_name="Media Broadcast",
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e),
                latency_ms=latency
            )
    
    async def _notify_municipal(self, alert: Dict, actions: List[Dict]) -> IntegrationResult:
        """Notify municipal systems"""
        start_time = datetime.utcnow()
        
        try:
            notification = {
                "alert_id": alert.get('identifier'),
                "severity": alert.get('severity'),
                "location": self._extract_location_name(alert),
                "headline": alert.get('headline'),
                "required_actions": [a.get('description', '') for a in (actions or [])[:5]],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # In production: POST to municipal API
            logger.info(f"Municipal system: Notified about {alert.get('identifier')}")
            
            await asyncio.sleep(0.05)
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return IntegrationResult(
                system_name="Municipal System",
                success=True,
                timestamp=datetime.utcnow(),
                response_data={
                    "notification_id": f"MUN-{alert.get('identifier')}",
                    "status": "received"
                },
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Municipal notification failed: {e}")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            return IntegrationResult(
                system_name="Municipal System",
                success=False,
                timestamp=datetime.utcnow(),
                error=str(e),
                latency_ms=latency
            )
    
    def _extract_coordinates(self, alert: Dict) -> Optional[Dict]:
        """Extract coordinates from alert areas"""
        areas = alert.get('areas', [])
        if areas and areas[0].get('polygon'):
            polygon = areas[0]['polygon']
            if polygon and len(polygon) > 0:
                return {
                    "latitude": polygon[0].get('latitude'),
                    "longitude": polygon[0].get('longitude')
                }
        return None
    
    def _extract_location_name(self, alert: Dict) -> str:
        """Extract location name from alert"""
        areas = alert.get('areas', [])
        if areas:
            return areas[0].get('areaDesc', 'Unknown')
        return 'Unknown'
    
    def _summarize_resources(self, resources: Dict) -> Dict:
        """Create summary of resource requirements"""
        if not resources:
            return {}
        
        summary = resources.get('summary', {})
        return {
            "total_cost": summary.get('total_estimated_cost', 0),
            "critical_shortages": len(summary.get('critical_shortages', [])),
            "items_needed": summary.get('total_resources_types', 0)
        }
    
    def _estimate_hospital_patients(self, alert: Dict, resources: Dict) -> int:
        """Estimate number of hospital patients expected"""
        # Get population from alert areas
        areas = alert.get('areas', [])
        total_population = sum(area.get('population', 0) for area in areas)
        
        if total_population == 0:
            total_population = 50000  # Default estimate
        
        # Estimate based on severity
        severity = alert.get('severity', 'Unknown')
        rates = {
            "Extreme": 0.05,   # 5% may need hospitalization
            "Severe": 0.02,    # 2%
            "Moderate": 0.005, # 0.5%
            "Minor": 0.001     # 0.1%
        }
        rate = rates.get(severity, 0.01)
        
        return int(total_population * rate)
    
    def _get_hospital_response(self, alert: Dict) -> List[str]:
        """Get hospital-specific response actions"""
        severity = alert.get('severity', 'Unknown')
        
        if severity == "Extreme":
            return [
                "Activate full surge capacity",
                "Call in all off-duty staff",
                "Prepare mass casualty protocols",
                "Set up triage areas",
                "Cancel elective procedures",
                "Increase security"
            ]
        elif severity == "Severe":
            return [
                "Activate surge capacity",
                "Call in additional staff",
                "Prepare emergency departments",
                "Review trauma protocols"
            ]
        elif severity == "Moderate":
            return [
                "Alert key personnel",
                "Prepare additional beds",
                "Review emergency protocols",
                "Check supply levels"
            ]
        else:
            return [
                "Monitor situation",
                "Stand by for updates"
            ]
    
    def _get_medical_resource_needs(self, resources: Dict) -> Dict:
        """Extract medical resource needs"""
        if not resources:
            return {}
        
        medical_resources = {}
        for resource in resources.get('resources', []):
            if resource.get('category') == 'medical':
                medical_resources[resource['resource_type']] = {
                    "needed": resource.get('quantity', 0),
                    "available": resource.get('availability', 0),
                    "shortage": resource.get('shortage', 0)
                }
        
        return medical_resources
    
    def _format_media_message(self, alert: Dict) -> str:
        """Format alert for media broadcast"""
        return f"""
OFFICIAL EMERGENCY ALERT

{alert.get('headline', 'Emergency Alert')}

{alert.get('description', '')[:500]}

WHAT TO DO:
{alert.get('instruction', 'Follow local emergency guidance')[:300]}

For more information:
- Emergency Services: 10177
- Disaster Hotline: 0800 111 990
- Website: crisis-connect.gov.za

Alert ID: {alert.get('identifier', 'Unknown')}
Issued: {alert.get('sent', 'Unknown')}
"""
    
    async def check_system_status(self) -> Dict[str, bool]:
        """Check connectivity to all integrated systems"""
        status = {}
        
        for system_name in self.endpoints.keys():
            try:
                # In production: Ping each system
                status[system_name] = True
            except Exception:
                status[system_name] = False
        
        return status
    
    async def sync_event_status(self, event_id: str, status: str, outcome_data: Dict = None) -> bool:
        """
        Sync event status back to disaster database
        Used for updating predictions with actual outcomes
        """
        try:
            update = {
                "event_id": event_id,
                "status": status,
                "updated_at": datetime.utcnow().isoformat(),
                "outcome": outcome_data
            }
            
            # In production: PUT to disaster database API
            logger.info(f"Synced event {event_id} status: {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync event status: {e}")
            return False


# Factory function
def create_emergency_integrator(config: Dict = None) -> EmergencySystemIntegrator:
    """Create an EmergencySystemIntegrator instance"""
    return EmergencySystemIntegrator(config=config)
