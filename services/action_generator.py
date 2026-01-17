# services/action_generator.py
"""
Action Generator Service - Phase 3

Generates specific, actionable emergency response actions based on predictions.
Each action includes stakeholder, priority, prerequisites, and success metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger("crisisconnect.action_generator")


class ActionPriority(Enum):
    """When should this action start?"""
    IMMEDIATE = 0       # Start now, no delay
    WITHIN_1HR = 60     # Start within 1 hour
    WITHIN_3HR = 180    # Start within 3 hours
    WITHIN_6HR = 360    # Start within 6 hours
    WITHIN_12HR = 720   # Start within 12 hours
    WITHIN_24HR = 1440  # Start within 24 hours


class ActionCategory(Enum):
    """Category of action"""
    COORDINATION = "Coordination"      # EOC activation, inter-agency coordination
    EVACUATION = "Evacuation"          # Evacuation orders and execution
    SHELTER = "Shelter"                # Emergency shelter operations
    RESCUE = "Rescue"                  # Search and rescue operations
    MEDICAL = "Medical"                # Medical preparedness and response
    INFRASTRUCTURE = "Infrastructure"  # Utilities, roads, critical infrastructure
    COMMUNICATION = "Communication"    # Public alerts, media, information
    LOGISTICS = "Logistics"            # Supplies, equipment, transport
    SECURITY = "Security"              # Law enforcement, access control
    RECOVERY = "Recovery"              # Post-disaster recovery actions


@dataclass
class Action:
    """Specific action to be taken in emergency response"""
    action_id: str
    description: str
    category: ActionCategory
    stakeholder: str              # Who should do this?
    priority: ActionPriority
    prerequisites: List[str]      # Action IDs that must complete first
    estimated_duration_min: int   # How long will this take?
    success_metric: str           # How do we know it's done?
    owner_contact: str            # Contact info for responsible party
    
    # Optional fields
    resources_needed: List[str] = field(default_factory=list)
    notes: str = ""
    is_critical: bool = False     # If true, failure blocks other actions
    
    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "description": self.description,
            "category": self.category.value,
            "stakeholder": self.stakeholder,
            "priority": self.priority.name,
            "priority_minutes": self.priority.value,
            "prerequisites": self.prerequisites,
            "estimated_duration_min": self.estimated_duration_min,
            "success_metric": self.success_metric,
            "owner_contact": self.owner_contact,
            "resources_needed": self.resources_needed,
            "notes": self.notes,
            "is_critical": self.is_critical
        }


@dataclass
class ActionPlan:
    """Complete action plan for emergency response"""
    plan_id: str
    hazard_type: str
    risk_level: str
    location: str
    created_at: datetime
    actions: List[Action]
    total_estimated_time_min: int
    critical_path: List[str]      # Sequence of critical actions
    
    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "hazard_type": self.hazard_type,
            "risk_level": self.risk_level,
            "location": self.location,
            "created_at": self.created_at.isoformat(),
            "actions": [a.to_dict() for a in self.actions],
            "total_actions": len(self.actions),
            "total_estimated_time_min": self.total_estimated_time_min,
            "critical_path": self.critical_path
        }
    
    def get_actions_by_priority(self) -> Dict[str, List[Action]]:
        """Group actions by priority"""
        grouped = {}
        for action in self.actions:
            key = action.priority.name
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(action)
        return grouped
    
    def get_actions_by_stakeholder(self) -> Dict[str, List[Action]]:
        """Group actions by stakeholder"""
        grouped = {}
        for action in self.actions:
            key = action.stakeholder
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(action)
        return grouped


class ActionGenerator:
    """
    Generate specific actions based on predictions
    
    Transforms abstract risk assessments into concrete action plans:
    - "Risk = HIGH, flood" → [Activate EOC, Issue evacuation, Open shelters, ...]
    
    Actions are:
    - Specific: Clear description of what to do
    - Assigned: Named stakeholder responsible
    - Timed: Priority and duration estimates
    - Measurable: Success metrics defined
    - Sequenced: Prerequisites identified
    """
    
    def __init__(self):
        self.plan_counter = 0
        
        # Default stakeholder contacts
        self.stakeholder_contacts = {
            "Disaster Management": "Disaster Management: +27 10 590 1000",
            "Municipality": "Municipal Manager: +27 11 555 0002",
            "Water Authority": "Water Authority: +27 11 555 0001",
            "Fire & Rescue": "Fire Chief: +27 11 555 0005",
            "Health": "Health Manager: +27 11 555 0006",
            "Social Development": "Social Development: +27 11 555 0004",
            "Traffic Authority": "Traffic Authority: +27 11 555 0007",
            "Communications": "Communications: +27 11 555 0003",
            "Police": "Police: 10111",
            "Military": "SANDF: +27 12 355 5555"
        }
        
        logger.info("ActionGenerator initialized")
    
    def generate_actions(self,
                        prediction: Dict,
                        location_data: Dict) -> ActionPlan:
        """
        Generate complete action plan for given prediction
        
        Args:
            prediction: Phase 2 prediction output
            location_data: Location information including population
            
        Returns:
            ActionPlan with all required actions
        """
        pred_data = prediction.get('prediction', {})
        risk_level = pred_data.get('risk_level', prediction.get('risk_level', 'LOW'))
        hours_to_peak = pred_data.get('hours_to_peak', prediction.get('hours_to_peak', 48))
        hazard_type = prediction.get('hazard_type', 'flood')
        location = prediction.get('location', location_data.get('name', 'Unknown'))
        
        actions = []
        
        # Generate hazard-specific actions
        if hazard_type.lower() == "flood":
            actions.extend(self._flood_actions(risk_level, hours_to_peak, location_data))
        elif hazard_type.lower() == "drought":
            actions.extend(self._drought_actions(risk_level, location_data))
        elif hazard_type.lower() == "storm":
            actions.extend(self._storm_actions(risk_level, hours_to_peak, location_data))
        else:
            # Generic disaster actions
            actions.extend(self._generic_disaster_actions(risk_level, hours_to_peak, location_data))
        
        # Add universal actions applicable to all hazards
        actions.extend(self._universal_actions(risk_level, hours_to_peak, location_data))
        
        # Adjust priorities based on time to peak
        actions = self._adjust_priorities(actions, hours_to_peak)
        
        # Identify and set prerequisites
        actions = self._set_prerequisites(actions)
        
        # Calculate critical path
        critical_path = self._calculate_critical_path(actions)
        
        # Create action plan
        self.plan_counter += 1
        plan_id = f"AP-{location.replace(' ', '_')}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self.plan_counter}"
        
        total_time = self._estimate_total_time(actions, critical_path)
        
        plan = ActionPlan(
            plan_id=plan_id,
            hazard_type=hazard_type,
            risk_level=risk_level,
            location=location,
            created_at=datetime.utcnow(),
            actions=actions,
            total_estimated_time_min=total_time,
            critical_path=critical_path
        )
        
        logger.info(f"Action plan generated: {plan_id}, {len(actions)} actions, risk={risk_level}")
        
        return plan
    
    def _flood_actions(self, risk_level: str, hours_to_peak: int, location_data: Dict) -> List[Action]:
        """Generate flood-specific actions"""
        actions = []
        population = location_data.get('population', 50000)
        
        if risk_level == "LOW":
            actions.extend([
                Action(
                    action_id="flood_monitor_1",
                    description="Activate enhanced flood monitoring - check water levels every 30 minutes",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Water Authority",
                    priority=ActionPriority.WITHIN_6HR,
                    prerequisites=[],
                    estimated_duration_min=0,  # Ongoing
                    success_metric="Monitoring data updated every 30 minutes in system",
                    owner_contact=self.stakeholder_contacts["Water Authority"]
                ),
                Action(
                    action_id="flood_review_1",
                    description="Review and verify emergency contact lists and evacuation routes",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Disaster Management",
                    priority=ActionPriority.WITHIN_12HR,
                    prerequisites=[],
                    estimated_duration_min=120,
                    success_metric="Contact lists verified, routes confirmed accessible",
                    owner_contact=self.stakeholder_contacts["Disaster Management"]
                )
            ])
        
        elif risk_level == "MODERATE":
            actions.extend([
                Action(
                    action_id="flood_prep_1",
                    description=f"Stage {max(5000, population // 10)} sandbags at {max(5, population // 10000)} key flood-prone locations",
                    category=ActionCategory.LOGISTICS,
                    stakeholder="Municipality",
                    priority=ActionPriority.WITHIN_6HR,
                    prerequisites=[],
                    estimated_duration_min=240,
                    success_metric="Sandbags positioned and photographed at all locations",
                    owner_contact=self.stakeholder_contacts["Municipality"],
                    resources_needed=["sandbags", "transport_vehicles", "personnel"]
                ),
                Action(
                    action_id="flood_prep_2",
                    description="Alert all pump operators - establish 2-hour standby readiness",
                    category=ActionCategory.INFRASTRUCTURE,
                    stakeholder="Water Authority",
                    priority=ActionPriority.WITHIN_3HR,
                    prerequisites=[],
                    estimated_duration_min=30,
                    success_metric="All operators acknowledged and confirmed on standby",
                    owner_contact=self.stakeholder_contacts["Water Authority"]
                ),
                Action(
                    action_id="flood_prep_3",
                    description="Issue public preparedness alert via SMS, radio, and social media",
                    category=ActionCategory.COMMUNICATION,
                    stakeholder="Communications",
                    priority=ActionPriority.WITHIN_1HR,
                    prerequisites=[],
                    estimated_duration_min=60,
                    success_metric="Alert distributed to all channels, reach metrics recorded",
                    owner_contact=self.stakeholder_contacts["Communications"]
                ),
                Action(
                    action_id="flood_prep_4",
                    description="Pre-position rescue boats at strategic access points",
                    category=ActionCategory.RESCUE,
                    stakeholder="Fire & Rescue",
                    priority=ActionPriority.WITHIN_3HR,
                    prerequisites=[],
                    estimated_duration_min=120,
                    success_metric="Boats positioned, crews assigned, equipment checked",
                    owner_contact=self.stakeholder_contacts["Fire & Rescue"],
                    resources_needed=["rescue_boats", "trained_crew"]
                )
            ])
        
        elif risk_level == "HIGH":
            actions.extend([
                Action(
                    action_id="flood_eoc_1",
                    description="Activate Emergency Operations Center - full staffing, all systems online",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Disaster Management",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=[],
                    estimated_duration_min=30,
                    success_metric="EOC operational, all stations manned, communications verified",
                    owner_contact=self.stakeholder_contacts["Disaster Management"],
                    is_critical=True
                ),
                Action(
                    action_id="flood_evac_1",
                    description="Issue precautionary evacuation advisory for flood zones A, B, C",
                    category=ActionCategory.EVACUATION,
                    stakeholder="Disaster Management",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=["flood_eoc_1"],
                    estimated_duration_min=30,
                    success_metric="Evacuation advisory published and distributed to all channels",
                    owner_contact=self.stakeholder_contacts["Disaster Management"],
                    is_critical=True
                ),
                Action(
                    action_id="flood_shelter_1",
                    description=f"Open {max(3, population // 15000)} emergency shelters, activate shelter staff",
                    category=ActionCategory.SHELTER,
                    stakeholder="Social Development",
                    priority=ActionPriority.WITHIN_1HR,
                    prerequisites=["flood_evac_1"],
                    estimated_duration_min=120,
                    success_metric=f"Shelters operational with {max(2000, population // 25)} beds ready",
                    owner_contact=self.stakeholder_contacts["Social Development"],
                    resources_needed=["shelter_beds", "food_supplies", "water", "medical_kits"],
                    is_critical=True
                ),
                Action(
                    action_id="flood_rescue_1",
                    description="Deploy rescue teams to high-risk zones, establish forward staging areas",
                    category=ActionCategory.RESCUE,
                    stakeholder="Fire & Rescue",
                    priority=ActionPriority.WITHIN_1HR,
                    prerequisites=["flood_eoc_1"],
                    estimated_duration_min=90,
                    success_metric="3+ rescue teams positioned, equipped, and in communication",
                    owner_contact=self.stakeholder_contacts["Fire & Rescue"],
                    resources_needed=["rescue_teams", "boats", "equipment"],
                    is_critical=True
                ),
                Action(
                    action_id="flood_medical_1",
                    description="Alert hospitals, activate emergency protocols, prepare for surge",
                    category=ActionCategory.MEDICAL,
                    stakeholder="Health",
                    priority=ActionPriority.WITHIN_1HR,
                    prerequisites=["flood_eoc_1"],
                    estimated_duration_min=60,
                    success_metric="All hospitals on alert, extra staff called in, supplies verified",
                    owner_contact=self.stakeholder_contacts["Health"]
                ),
                Action(
                    action_id="flood_transport_1",
                    description="Mobilize evacuation transport - buses, municipal vehicles",
                    category=ActionCategory.LOGISTICS,
                    stakeholder="Municipality",
                    priority=ActionPriority.WITHIN_1HR,
                    prerequisites=["flood_evac_1"],
                    estimated_duration_min=90,
                    success_metric=f"{max(20, population // 2500)} vehicles ready at designated pickup points",
                    owner_contact=self.stakeholder_contacts["Municipality"],
                    resources_needed=["buses", "drivers"]
                )
            ])
        
        else:  # CRITICAL
            actions.extend([
                Action(
                    action_id="flood_critical_eoc",
                    description="EMERGENCY: Full EOC activation with all agency heads present",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Disaster Management",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=[],
                    estimated_duration_min=15,
                    success_metric="EOC at full capacity, all agency heads connected",
                    owner_contact=self.stakeholder_contacts["Disaster Management"],
                    is_critical=True
                ),
                Action(
                    action_id="flood_critical_evac",
                    description="MANDATORY EVACUATION: Issue immediate evacuation order for all flood zones",
                    category=ActionCategory.EVACUATION,
                    stakeholder="Disaster Management",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=[],
                    estimated_duration_min=15,
                    success_metric="Mandatory evacuation order issued, sirens activated",
                    owner_contact=self.stakeholder_contacts["Disaster Management"],
                    is_critical=True
                ),
                Action(
                    action_id="flood_critical_roads",
                    description="Close all roads entering flood zones, establish traffic control",
                    category=ActionCategory.SECURITY,
                    stakeholder="Traffic Authority",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=[],
                    estimated_duration_min=60,
                    success_metric="All entry roads closed, barriers installed, officers posted",
                    owner_contact=self.stakeholder_contacts["Traffic Authority"],
                    is_critical=True
                ),
                Action(
                    action_id="flood_critical_rescue",
                    description="Maximum rescue deployment - all available teams on 24-hour shifts",
                    category=ActionCategory.RESCUE,
                    stakeholder="Fire & Rescue",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=[],
                    estimated_duration_min=60,
                    success_metric="All rescue units deployed, mutual aid activated if needed",
                    owner_contact=self.stakeholder_contacts["Fire & Rescue"],
                    is_critical=True
                ),
                Action(
                    action_id="flood_critical_military",
                    description="Request military assistance for evacuation and rescue operations",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Disaster Management",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=["flood_critical_eoc"],
                    estimated_duration_min=30,
                    success_metric="Military assistance formally requested, ETA confirmed",
                    owner_contact=self.stakeholder_contacts["Military"]
                ),
                Action(
                    action_id="flood_critical_utilities",
                    description="Coordinate utility shutoffs in flood zones to prevent electrocution",
                    category=ActionCategory.INFRASTRUCTURE,
                    stakeholder="Municipality",
                    priority=ActionPriority.WITHIN_1HR,
                    prerequisites=["flood_critical_evac"],
                    estimated_duration_min=120,
                    success_metric="Power safely disconnected in evacuation zones",
                    owner_contact=self.stakeholder_contacts["Municipality"],
                    is_critical=True
                )
            ])
        
        return actions
    
    def _drought_actions(self, risk_level: str, location_data: Dict) -> List[Action]:
        """Generate drought-specific actions"""
        actions = []
        population = location_data.get('population', 50000)
        
        if risk_level in ["LOW", "MODERATE"]:
            actions.extend([
                Action(
                    action_id="drought_monitor_1",
                    description="Increase water level monitoring frequency to daily",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Water Authority",
                    priority=ActionPriority.WITHIN_24HR,
                    prerequisites=[],
                    estimated_duration_min=0,
                    success_metric="Daily water level reports in system",
                    owner_contact=self.stakeholder_contacts["Water Authority"]
                ),
                Action(
                    action_id="drought_awareness_1",
                    description="Launch water conservation awareness campaign",
                    category=ActionCategory.COMMUNICATION,
                    stakeholder="Communications",
                    priority=ActionPriority.WITHIN_24HR,
                    prerequisites=[],
                    estimated_duration_min=240,
                    success_metric="Campaign materials distributed, media coverage secured",
                    owner_contact=self.stakeholder_contacts["Communications"]
                )
            ])
        
        if risk_level in ["HIGH", "CRITICAL"]:
            actions.extend([
                Action(
                    action_id="drought_restrict_1",
                    description="Implement water restrictions - Level 2 (no outdoor watering)",
                    category=ActionCategory.INFRASTRUCTURE,
                    stakeholder="Water Authority",
                    priority=ActionPriority.WITHIN_3HR,
                    prerequisites=[],
                    estimated_duration_min=60,
                    success_metric="Restrictions announced, enforcement plan activated",
                    owner_contact=self.stakeholder_contacts["Water Authority"],
                    is_critical=True
                ),
                Action(
                    action_id="drought_supply_1",
                    description=f"Establish {max(5, population // 10000)} emergency water distribution points",
                    category=ActionCategory.LOGISTICS,
                    stakeholder="Municipality",
                    priority=ActionPriority.WITHIN_6HR,
                    prerequisites=[],
                    estimated_duration_min=180,
                    success_metric="Distribution points operational, schedules published",
                    owner_contact=self.stakeholder_contacts["Municipality"],
                    resources_needed=["water_tankers", "distribution_equipment"]
                ),
                Action(
                    action_id="drought_agriculture_1",
                    description="Coordinate with agricultural sector on emergency water allocation",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Water Authority",
                    priority=ActionPriority.WITHIN_12HR,
                    prerequisites=[],
                    estimated_duration_min=180,
                    success_metric="Agricultural water allocation plan agreed and communicated",
                    owner_contact=self.stakeholder_contacts["Water Authority"]
                )
            ])
        
        return actions
    
    def _storm_actions(self, risk_level: str, hours_to_peak: int, location_data: Dict) -> List[Action]:
        """Generate storm-specific actions"""
        actions = []
        population = location_data.get('population', 50000)
        
        if risk_level == "LOW":
            actions.append(
                Action(
                    action_id="storm_monitor_1",
                    description="Activate storm tracking and monitoring protocols",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Disaster Management",
                    priority=ActionPriority.WITHIN_12HR,
                    prerequisites=[],
                    estimated_duration_min=0,
                    success_metric="Storm tracking active, updates every 3 hours",
                    owner_contact=self.stakeholder_contacts["Disaster Management"]
                )
            )
        
        elif risk_level == "MODERATE":
            actions.extend([
                Action(
                    action_id="storm_secure_1",
                    description="Issue advisory to secure loose outdoor items and structures",
                    category=ActionCategory.COMMUNICATION,
                    stakeholder="Communications",
                    priority=ActionPriority.WITHIN_3HR,
                    prerequisites=[],
                    estimated_duration_min=60,
                    success_metric="Advisory distributed via all channels",
                    owner_contact=self.stakeholder_contacts["Communications"]
                ),
                Action(
                    action_id="storm_power_1",
                    description="Alert power utility to prepare for outages, pre-position repair crews",
                    category=ActionCategory.INFRASTRUCTURE,
                    stakeholder="Municipality",
                    priority=ActionPriority.WITHIN_6HR,
                    prerequisites=[],
                    estimated_duration_min=120,
                    success_metric="Repair crews on standby, emergency equipment ready",
                    owner_contact=self.stakeholder_contacts["Municipality"]
                )
            ])
        
        elif risk_level in ["HIGH", "CRITICAL"]:
            actions.extend([
                Action(
                    action_id="storm_eoc_1",
                    description="Activate Emergency Operations Center for storm response",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Disaster Management",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=[],
                    estimated_duration_min=30,
                    success_metric="EOC operational, weather feeds active",
                    owner_contact=self.stakeholder_contacts["Disaster Management"],
                    is_critical=True
                ),
                Action(
                    action_id="storm_shelter_1",
                    description="Open storm shelters for residents in vulnerable structures",
                    category=ActionCategory.SHELTER,
                    stakeholder="Social Development",
                    priority=ActionPriority.WITHIN_1HR,
                    prerequisites=["storm_eoc_1"],
                    estimated_duration_min=90,
                    success_metric="Shelters open, locations broadcast to public",
                    owner_contact=self.stakeholder_contacts["Social Development"],
                    resources_needed=["shelter_supplies"]
                ),
                Action(
                    action_id="storm_trees_1",
                    description="Pre-position tree removal and debris clearing equipment",
                    category=ActionCategory.INFRASTRUCTURE,
                    stakeholder="Municipality",
                    priority=ActionPriority.WITHIN_3HR,
                    prerequisites=[],
                    estimated_duration_min=120,
                    success_metric="Equipment staged at strategic locations",
                    owner_contact=self.stakeholder_contacts["Municipality"],
                    resources_needed=["chainsaws", "trucks", "crews"]
                )
            ])
        
        return actions
    
    def _generic_disaster_actions(self, risk_level: str, hours_to_peak: int, location_data: Dict) -> List[Action]:
        """Generate generic disaster response actions"""
        actions = []
        
        if risk_level in ["HIGH", "CRITICAL"]:
            actions.extend([
                Action(
                    action_id="generic_eoc_1",
                    description="Activate Emergency Operations Center",
                    category=ActionCategory.COORDINATION,
                    stakeholder="Disaster Management",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=[],
                    estimated_duration_min=30,
                    success_metric="EOC operational",
                    owner_contact=self.stakeholder_contacts["Disaster Management"],
                    is_critical=True
                ),
                Action(
                    action_id="generic_alert_1",
                    description="Issue public safety alert through all channels",
                    category=ActionCategory.COMMUNICATION,
                    stakeholder="Communications",
                    priority=ActionPriority.IMMEDIATE,
                    prerequisites=[],
                    estimated_duration_min=30,
                    success_metric="Alert distributed",
                    owner_contact=self.stakeholder_contacts["Communications"]
                )
            ])
        
        return actions
    
    def _universal_actions(self, risk_level: str, hours_to_peak: int, location_data: Dict) -> List[Action]:
        """Actions that apply to all hazard types"""
        actions = []
        
        if risk_level in ["HIGH", "CRITICAL"]:
            actions.extend([
                Action(
                    action_id="universal_media_1",
                    description="Establish media briefing schedule - updates every 2 hours",
                    category=ActionCategory.COMMUNICATION,
                    stakeholder="Communications",
                    priority=ActionPriority.WITHIN_1HR,
                    prerequisites=[],
                    estimated_duration_min=30,
                    success_metric="Media briefing schedule published, first briefing completed",
                    owner_contact=self.stakeholder_contacts["Communications"]
                ),
                Action(
                    action_id="universal_vulnerable_1",
                    description="Activate welfare checks for registered vulnerable residents",
                    category=ActionCategory.MEDICAL,
                    stakeholder="Social Development",
                    priority=ActionPriority.WITHIN_3HR,
                    prerequisites=[],
                    estimated_duration_min=240,
                    success_metric="All registered vulnerable residents contacted",
                    owner_contact=self.stakeholder_contacts["Social Development"]
                ),
                Action(
                    action_id="universal_hotline_1",
                    description="Activate emergency information hotline with extra operators",
                    category=ActionCategory.COMMUNICATION,
                    stakeholder="Communications",
                    priority=ActionPriority.WITHIN_1HR,
                    prerequisites=[],
                    estimated_duration_min=60,
                    success_metric="Hotline operational with <2 min average wait time",
                    owner_contact=self.stakeholder_contacts["Communications"]
                )
            ])
        
        if risk_level == "CRITICAL":
            actions.append(
                Action(
                    action_id="universal_curfew_1",
                    description="Consider implementing emergency curfew in affected areas",
                    category=ActionCategory.SECURITY,
                    stakeholder="Police",
                    priority=ActionPriority.WITHIN_3HR,
                    prerequisites=[],
                    estimated_duration_min=60,
                    success_metric="Curfew decision made and communicated if implemented",
                    owner_contact=self.stakeholder_contacts["Police"],
                    notes="Requires mayoral/provincial authorization"
                )
            )
        
        return actions
    
    def _adjust_priorities(self, actions: List[Action], hours_to_peak: int) -> List[Action]:
        """Adjust action priorities based on time available"""
        
        # If less than 6 hours to peak, escalate priorities
        if hours_to_peak < 6:
            for action in actions:
                if action.priority in [ActionPriority.WITHIN_6HR, ActionPriority.WITHIN_12HR, ActionPriority.WITHIN_24HR]:
                    action.priority = ActionPriority.WITHIN_1HR
                elif action.priority == ActionPriority.WITHIN_3HR:
                    action.priority = ActionPriority.IMMEDIATE
        
        # If less than 12 hours, moderate escalation
        elif hours_to_peak < 12:
            for action in actions:
                if action.priority in [ActionPriority.WITHIN_12HR, ActionPriority.WITHIN_24HR]:
                    action.priority = ActionPriority.WITHIN_6HR
                elif action.priority == ActionPriority.WITHIN_6HR:
                    action.priority = ActionPriority.WITHIN_3HR
        
        return actions
    
    def _set_prerequisites(self, actions: List[Action]) -> List[Action]:
        """Identify and set action prerequisites"""
        
        # Find EOC activation actions
        eoc_actions = [a for a in actions if "eoc" in a.action_id.lower() or "activate" in a.description.lower()]
        eoc_id = eoc_actions[0].action_id if eoc_actions else None
        
        if eoc_id:
            # Most operational actions require EOC to be active
            for action in actions:
                if action.action_id != eoc_id:
                    # Categories that typically need EOC first
                    if action.category in [ActionCategory.EVACUATION, ActionCategory.SHELTER, 
                                          ActionCategory.RESCUE] and not action.prerequisites:
                        if action.stakeholder not in ["Communications", "Municipality"]:
                            action.prerequisites.append(eoc_id)
        
        # Evacuation must precede shelter operations
        evac_actions = [a for a in actions if a.category == ActionCategory.EVACUATION]
        if evac_actions:
            evac_id = evac_actions[0].action_id
            for action in actions:
                if action.category == ActionCategory.SHELTER and evac_id not in action.prerequisites:
                    if action.action_id != evac_id:
                        action.prerequisites.append(evac_id)
        
        return actions
    
    def _calculate_critical_path(self, actions: List[Action]) -> List[str]:
        """Calculate the critical path through actions"""
        critical_actions = [a for a in actions if a.is_critical]
        
        # Sort by priority (most urgent first)
        critical_actions.sort(key=lambda a: a.priority.value)
        
        # Build path respecting prerequisites
        path = []
        completed = set()
        
        while len(path) < len(critical_actions):
            for action in critical_actions:
                if action.action_id in completed:
                    continue
                # Check if all prerequisites are met
                if all(prereq in completed for prereq in action.prerequisites):
                    path.append(action.action_id)
                    completed.add(action.action_id)
                    break
            else:
                # No action could be added - break to avoid infinite loop
                break
        
        return path
    
    def _estimate_total_time(self, actions: List[Action], critical_path: List[str]) -> int:
        """Estimate total time to complete critical path"""
        total = 0
        for action_id in critical_path:
            action = next((a for a in actions if a.action_id == action_id), None)
            if action:
                total += action.estimated_duration_min
        return total
    
    def get_actions_for_stakeholder(self, 
                                    action_plan: ActionPlan, 
                                    stakeholder: str) -> List[Action]:
        """Get all actions assigned to a specific stakeholder"""
        return [a for a in action_plan.actions if a.stakeholder == stakeholder]
    
    def get_immediate_actions(self, action_plan: ActionPlan) -> List[Action]:
        """Get all actions that should start immediately"""
        return [a for a in action_plan.actions if a.priority == ActionPriority.IMMEDIATE]


# Factory function
def create_action_generator() -> ActionGenerator:
    """Create an ActionGenerator instance"""
    return ActionGenerator()
