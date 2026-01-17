# services/resource_calculator.py
"""
Resource Calculator Service - Phase 3

Calculates resources needed for disaster response based on:
- Population affected
- Risk level and hazard type
- Action plan requirements
- Humanitarian standards (Sphere standards)

Provides gap analysis against current inventory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger("crisisconnect.resource_calculator")


@dataclass
class ResourceRequirement:
    """Single resource requirement with availability analysis"""
    resource_type: str          # e.g., shelter_beds, sandbags, rescue_teams
    category: str               # e.g., shelter, food, water, medical, transport
    quantity: int               # How many needed
    unit: str                   # Unit of measurement
    estimated_cost_per_unit: float
    total_estimated_cost: float
    availability: int           # Currently available
    shortage: int               # How many we need but don't have
    source: str                 # Where to get them
    lead_time_minutes: int      # How long to acquire/deploy
    priority: str               # critical, high, medium, low
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "resource_type": self.resource_type,
            "category": self.category,
            "quantity": self.quantity,
            "unit": self.unit,
            "estimated_cost_per_unit": self.estimated_cost_per_unit,
            "total_estimated_cost": self.total_estimated_cost,
            "availability": self.availability,
            "shortage": self.shortage,
            "shortage_percentage": (self.shortage / self.quantity * 100) if self.quantity > 0 else 0,
            "source": self.source,
            "lead_time_minutes": self.lead_time_minutes,
            "priority": self.priority,
            "notes": self.notes
        }


@dataclass
class ResourceReport:
    """Complete resource requirements report"""
    report_id: str
    location: str
    hazard_type: str
    risk_level: str
    population_affected: int
    created_at: datetime
    
    # Resource lists by category
    resources: List[ResourceRequirement]
    
    # Summary metrics
    total_estimated_cost: float
    total_shortage_cost: float
    critical_shortages: List[Dict]
    
    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "location": self.location,
            "hazard_type": self.hazard_type,
            "risk_level": self.risk_level,
            "population_affected": self.population_affected,
            "created_at": self.created_at.isoformat(),
            "resources": [r.to_dict() for r in self.resources],
            "summary": {
                "total_resources_types": len(self.resources),
                "total_estimated_cost": self.total_estimated_cost,
                "total_shortage_cost": self.total_shortage_cost,
                "items_with_shortage": len([r for r in self.resources if r.shortage > 0]),
                "critical_shortages": self.critical_shortages
            }
        }
    
    def get_resources_by_category(self) -> Dict[str, List[ResourceRequirement]]:
        """Group resources by category"""
        grouped = {}
        for resource in self.resources:
            if resource.category not in grouped:
                grouped[resource.category] = []
            grouped[resource.category].append(resource)
        return grouped


class ResourceCalculator:
    """
    Calculate resources needed for disaster response
    
    Based on:
    - Sphere Humanitarian Standards
    - Population affected
    - Risk level and hazard type
    - Historical disaster response data
    
    Provides:
    - Quantity estimates
    - Cost estimates
    - Availability analysis
    - Gap identification
    - Sourcing recommendations
    """
    
    def __init__(self):
        self.report_counter = 0
        
        # Sphere standards - minimum humanitarian requirements
        # Reference: https://spherestandards.org/
        self.sphere_standards = {
            "water_liters_per_person_per_day": 20,  # Minimum for survival
            "shelter_space_sqm_per_person": 3.5,    # Minimum covered area
            "food_kcal_per_person_per_day": 2100,   # Minimum nutrition
            "latrine_ratio": 20,                     # 1 latrine per 20 people
            "health_workers_per_10000": 22          # Minimum health staffing
        }
        
        # Default inventory (in production, query from inventory system)
        self.default_inventory = {
            "shelter_beds": 5000,
            "drinking_water_liters": 100000,
            "food_meals": 15000,
            "rescue_teams": 5,
            "rescue_boats": 10,
            "sandbags": 20000,
            "medical_personnel": 100,
            "ambulances": 20,
            "communications_units": 30,
            "generators": 15,
            "water_pumps": 25,
            "transport_buses": 50,
            "blankets": 10000,
            "hygiene_kits": 5000,
            "first_aid_kits": 2000,
            "water_purification_tablets": 50000,
            "tarpaulins": 3000,
            "portable_toilets": 100
        }
        
        # Cost estimates (ZAR - South African Rand)
        self.cost_estimates = {
            "shelter_beds": 150,              # Per bed per day
            "drinking_water_liters": 0.50,    # Per liter
            "food_meals": 35,                 # Per meal
            "rescue_teams": 15000,            # Per team per day
            "rescue_boats": 5000,             # Per boat per day
            "sandbags": 5,                    # Per bag
            "medical_personnel": 2500,        # Per person per day
            "ambulances": 3000,               # Per vehicle per day
            "communications_units": 500,      # Per unit per day
            "generators": 1500,               # Per unit per day
            "water_pumps": 800,               # Per pump per day
            "transport_buses": 4000,          # Per bus per day
            "blankets": 150,                  # Per blanket
            "hygiene_kits": 120,              # Per kit
            "first_aid_kits": 250,            # Per kit
            "water_purification_tablets": 2,  # Per tablet
            "tarpaulins": 350,                # Per tarp
            "portable_toilets": 2000          # Per unit per day
        }
        
        # Lead times in minutes
        self.lead_times = {
            "shelter_beds": 120,
            "drinking_water_liters": 180,
            "food_meals": 240,
            "rescue_teams": 60,
            "rescue_boats": 90,
            "sandbags": 120,
            "medical_personnel": 90,
            "ambulances": 30,
            "communications_units": 60,
            "generators": 120,
            "water_pumps": 90,
            "transport_buses": 60,
            "blankets": 120,
            "hygiene_kits": 180,
            "first_aid_kits": 60,
            "water_purification_tablets": 120,
            "tarpaulins": 120,
            "portable_toilets": 180
        }
        
        # Sources for resources
        self.sources = {
            "shelter_beds": "Emergency Shelters, Hotels, Schools, Community Centers",
            "drinking_water_liters": "Water Authority, Bottling Plants, Municipal Reserves",
            "food_meals": "Catering Companies, NGOs, Food Banks, Military",
            "rescue_teams": "Fire & Rescue, Military, NGO Partners, Mutual Aid",
            "rescue_boats": "Fire & Rescue, Maritime Authority, Private Hire",
            "sandbags": "Municipal Stores, Sand Suppliers, Hardware Stores",
            "medical_personnel": "Hospitals, Private Clinics, Medical NGOs",
            "ambulances": "EMS Services, Private Ambulance Companies",
            "communications_units": "Government Comms, Telecom Providers",
            "generators": "Municipal Stores, Equipment Rental Companies",
            "water_pumps": "Water Authority, Fire Services, Equipment Rental",
            "transport_buses": "Public Transport, School Buses, Private Hire",
            "blankets": "Social Development, NGOs, Retail Emergency Stock",
            "hygiene_kits": "Health Department, NGOs, Procurement",
            "first_aid_kits": "Health Department, Pharmacies, NGOs",
            "water_purification_tablets": "Health Department, NGOs, Pharmacies",
            "tarpaulins": "Municipal Stores, Hardware Suppliers",
            "portable_toilets": "Sanitation Services, Event Companies"
        }
        
        logger.info("ResourceCalculator initialized")
    
    def calculate_resources(self,
                           prediction: Dict,
                           location_data: Dict,
                           actions: List[Dict] = None) -> ResourceReport:
        """
        Calculate all resources needed for disaster response
        
        Args:
            prediction: Phase 2 prediction output
            location_data: Location information including population
            actions: Optional list of actions to consider for resources
            
        Returns:
            ResourceReport with all requirements and gap analysis
        """
        pred_data = prediction.get('prediction', {})
        risk_level = pred_data.get('risk_level', prediction.get('risk_level', 'LOW'))
        hazard_type = prediction.get('hazard_type', 'flood')
        location = prediction.get('location', location_data.get('name', 'Unknown'))
        population = location_data.get('population', 50000)
        
        resources = []
        
        # Calculate hazard-specific resources
        if hazard_type.lower() == "flood":
            resources.extend(self._flood_resources(risk_level, population))
        elif hazard_type.lower() == "drought":
            resources.extend(self._drought_resources(risk_level, population))
        elif hazard_type.lower() == "storm":
            resources.extend(self._storm_resources(risk_level, population))
        
        # Add universal resources
        resources.extend(self._universal_resources(risk_level, population))
        
        # Calculate gaps
        resources = self._calculate_gaps(resources)
        
        # Identify critical shortages
        critical_shortages = self._identify_critical_shortages(resources)
        
        # Calculate totals
        total_cost = sum(r.total_estimated_cost for r in resources)
        shortage_cost = sum(r.shortage * r.estimated_cost_per_unit for r in resources if r.shortage > 0)
        
        # Create report
        self.report_counter += 1
        report_id = f"RR-{location.replace(' ', '_')}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self.report_counter}"
        
        report = ResourceReport(
            report_id=report_id,
            location=location,
            hazard_type=hazard_type,
            risk_level=risk_level,
            population_affected=self._estimate_affected_population(population, risk_level),
            created_at=datetime.utcnow(),
            resources=resources,
            total_estimated_cost=total_cost,
            total_shortage_cost=shortage_cost,
            critical_shortages=critical_shortages
        )
        
        logger.info(f"Resource report generated: {report_id}, {len(resources)} resource types, total cost: R{total_cost:,.2f}")
        
        return report
    
    def _estimate_affected_population(self, total_population: int, risk_level: str) -> int:
        """Estimate population that will need assistance"""
        # Percentage of population affected by risk level
        affected_rates = {
            "LOW": 0.02,       # 2% may need some assistance
            "MODERATE": 0.10,  # 10% affected
            "HIGH": 0.25,      # 25% affected
            "CRITICAL": 0.50   # 50% affected
        }
        rate = affected_rates.get(risk_level, 0.10)
        return int(total_population * rate)
    
    def _flood_resources(self, risk_level: str, population: int) -> List[ResourceRequirement]:
        """Calculate flood-specific resources"""
        resources = []
        affected = self._estimate_affected_population(population, risk_level)
        
        # Evacuation rate by risk level
        evacuation_rates = {
            "LOW": 0,
            "MODERATE": 0.05,
            "HIGH": 0.20,
            "CRITICAL": 0.40
        }
        evac_rate = evacuation_rates.get(risk_level, 0.10)
        people_evacuated = int(population * evac_rate)
        
        if people_evacuated > 0:
            # Shelter beds: 1 bed per 3 evacuees (families share)
            shelter_beds = max(100, int(people_evacuated / 3))
            resources.append(self._create_resource(
                "shelter_beds", "shelter", shelter_beds, "beds",
                priority="critical" if risk_level in ["HIGH", "CRITICAL"] else "high"
            ))
            
            # Blankets: 2 per evacuee
            blankets = people_evacuated * 2
            resources.append(self._create_resource(
                "blankets", "shelter", blankets, "blankets",
                priority="high"
            ))
        
        # Water: Sphere standard 20L/person/day for 7 days
        days = 7
        water_liters = affected * self.sphere_standards["water_liters_per_person_per_day"] * days
        resources.append(self._create_resource(
            "drinking_water_liters", "water", water_liters, "liters",
            priority="critical"
        ))
        
        # Food: 3 meals/person/day for 7 days
        meals = affected * 3 * days
        resources.append(self._create_resource(
            "food_meals", "food", meals, "meals",
            priority="critical"
        ))
        
        # Rescue teams: 1 team per 5,000 people in critical zone
        rescue_teams = max(3, population // 5000) if risk_level in ["HIGH", "CRITICAL"] else 1
        resources.append(self._create_resource(
            "rescue_teams", "rescue", rescue_teams, "teams",
            priority="critical" if risk_level in ["HIGH", "CRITICAL"] else "medium"
        ))
        
        # Rescue boats
        boats = max(2, rescue_teams * 2) if risk_level in ["HIGH", "CRITICAL"] else 0
        if boats > 0:
            resources.append(self._create_resource(
                "rescue_boats", "rescue", boats, "boats",
                priority="critical"
            ))
        
        # Sandbags: Based on flood zone perimeter
        if risk_level != "LOW":
            sandbags = max(1000, int(population / 50))
            resources.append(self._create_resource(
                "sandbags", "infrastructure", sandbags, "bags",
                priority="high" if risk_level in ["HIGH", "CRITICAL"] else "medium"
            ))
        
        # Water pumps for drainage
        if risk_level in ["HIGH", "CRITICAL"]:
            pumps = max(5, population // 10000)
            resources.append(self._create_resource(
                "water_pumps", "infrastructure", pumps, "pumps",
                priority="high"
            ))
        
        # Hygiene kits: 1 per evacuated household (assume 4 per household)
        if people_evacuated > 0:
            hygiene_kits = max(100, people_evacuated // 4)
            resources.append(self._create_resource(
                "hygiene_kits", "health", hygiene_kits, "kits",
                priority="medium"
            ))
        
        # Portable toilets: Sphere standard 1 per 20 people
        if people_evacuated > 0:
            toilets = max(10, people_evacuated // self.sphere_standards["latrine_ratio"])
            resources.append(self._create_resource(
                "portable_toilets", "sanitation", toilets, "units",
                priority="high"
            ))
        
        return resources
    
    def _drought_resources(self, risk_level: str, population: int) -> List[ResourceRequirement]:
        """Calculate drought-specific resources"""
        resources = []
        affected = self._estimate_affected_population(population, risk_level)
        
        # Water is the primary need
        # Drought requires longer-term water supply
        days = 30 if risk_level in ["HIGH", "CRITICAL"] else 14
        water_liters = affected * self.sphere_standards["water_liters_per_person_per_day"] * days
        resources.append(self._create_resource(
            "drinking_water_liters", "water", water_liters, "liters",
            priority="critical"
        ))
        
        # Water purification tablets for extended supply
        tablets = affected * days * 2  # 2 tablets per person per day
        resources.append(self._create_resource(
            "water_purification_tablets", "water", tablets, "tablets",
            priority="high"
        ))
        
        # Water tankers/transport
        if risk_level in ["HIGH", "CRITICAL"]:
            # Assume need to distribute water to affected areas
            transport_buses = max(5, affected // 5000)  # For water tanker trips
            resources.append(self._create_resource(
                "transport_buses", "transport", transport_buses, "vehicles",
                priority="high",
                notes="For water distribution"
            ))
        
        return resources
    
    def _storm_resources(self, risk_level: str, population: int) -> List[ResourceRequirement]:
        """Calculate storm-specific resources"""
        resources = []
        affected = self._estimate_affected_population(population, risk_level)
        
        # Shelter for those in vulnerable structures
        if risk_level in ["HIGH", "CRITICAL"]:
            shelter_rate = 0.15 if risk_level == "HIGH" else 0.30
            people_sheltered = int(population * shelter_rate)
            shelter_beds = max(200, people_sheltered // 3)
            resources.append(self._create_resource(
                "shelter_beds", "shelter", shelter_beds, "beds",
                priority="critical"
            ))
            
            blankets = people_sheltered * 2
            resources.append(self._create_resource(
                "blankets", "shelter", blankets, "blankets",
                priority="high"
            ))
        
        # Generators for power outages
        generators = max(10, population // 5000) if risk_level != "LOW" else 5
        resources.append(self._create_resource(
            "generators", "infrastructure", generators, "units",
            priority="high" if risk_level in ["HIGH", "CRITICAL"] else "medium"
        ))
        
        # Tarpaulins for temporary roof repairs
        if risk_level in ["HIGH", "CRITICAL"]:
            tarps = max(500, affected // 10)
            resources.append(self._create_resource(
                "tarpaulins", "shelter", tarps, "tarps",
                priority="high"
            ))
        
        # Communications equipment
        if risk_level in ["HIGH", "CRITICAL"]:
            comms = max(20, population // 10000)
            resources.append(self._create_resource(
                "communications_units", "communications", comms, "units",
                priority="high"
            ))
        
        return resources
    
    def _universal_resources(self, risk_level: str, population: int) -> List[ResourceRequirement]:
        """Resources needed for all disaster types"""
        resources = []
        affected = self._estimate_affected_population(population, risk_level)
        
        if risk_level in ["HIGH", "CRITICAL"]:
            # Medical personnel: Sphere standard ~22 per 10,000
            medical_staff = max(10, int(affected / 10000 * self.sphere_standards["health_workers_per_10000"]))
            resources.append(self._create_resource(
                "medical_personnel", "medical", medical_staff, "people",
                priority="critical"
            ))
            
            # Ambulances
            ambulances = max(5, affected // 5000)
            resources.append(self._create_resource(
                "ambulances", "medical", ambulances, "vehicles",
                priority="critical"
            ))
            
            # First aid kits
            first_aid = max(100, affected // 50)
            resources.append(self._create_resource(
                "first_aid_kits", "medical", first_aid, "kits",
                priority="high"
            ))
            
            # Transport buses for evacuation
            buses = max(10, affected // 500)
            resources.append(self._create_resource(
                "transport_buses", "transport", buses, "buses",
                priority="high"
            ))
            
            # Communications units
            comms = max(15, population // 10000)
            resources.append(self._create_resource(
                "communications_units", "communications", comms, "units",
                priority="high"
            ))
        
        return resources
    
    def _create_resource(self,
                        resource_type: str,
                        category: str,
                        quantity: int,
                        unit: str,
                        priority: str = "medium",
                        notes: str = "") -> ResourceRequirement:
        """Create a ResourceRequirement with defaults"""
        cost_per_unit = self.cost_estimates.get(resource_type, 100)
        
        return ResourceRequirement(
            resource_type=resource_type,
            category=category,
            quantity=quantity,
            unit=unit,
            estimated_cost_per_unit=cost_per_unit,
            total_estimated_cost=quantity * cost_per_unit,
            availability=self._get_availability(resource_type),
            shortage=0,  # Will be calculated
            source=self.sources.get(resource_type, "To be determined"),
            lead_time_minutes=self.lead_times.get(resource_type, 120),
            priority=priority,
            notes=notes
        )
    
    def _get_availability(self, resource_type: str) -> int:
        """
        Get current availability of resource
        In production: Query inventory management system
        """
        return self.default_inventory.get(resource_type, 0)
    
    def _calculate_gaps(self, resources: List[ResourceRequirement]) -> List[ResourceRequirement]:
        """Calculate shortage for each resource"""
        for resource in resources:
            shortage = max(0, resource.quantity - resource.availability)
            resource.shortage = shortage
            
            if shortage > 0:
                shortage_pct = (shortage / resource.quantity) * 100
                logger.warning(
                    f"SHORTAGE: {resource.resource_type} - need {shortage} more "
                    f"(have {resource.availability}, need {resource.quantity}, {shortage_pct:.1f}% short)"
                )
        
        return resources
    
    def _identify_critical_shortages(self, resources: List[ResourceRequirement]) -> List[Dict]:
        """Identify resources with critical shortages"""
        critical = []
        
        for r in resources:
            if r.shortage > 0:
                shortage_pct = (r.shortage / r.quantity) * 100
                
                # Critical if: priority is critical/high AND shortage > 20%
                if (r.priority in ["critical", "high"] and shortage_pct > 20) or shortage_pct > 50:
                    critical.append({
                        "resource": r.resource_type,
                        "category": r.category,
                        "needed": r.quantity,
                        "available": r.availability,
                        "shortage": r.shortage,
                        "shortage_percentage": shortage_pct,
                        "source": r.source,
                        "lead_time_minutes": r.lead_time_minutes,
                        "estimated_cost_to_fill": r.shortage * r.estimated_cost_per_unit,
                        "priority": r.priority
                    })
        
        # Sort by priority and shortage percentage
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        critical.sort(key=lambda x: (priority_order.get(x["priority"], 4), -x["shortage_percentage"]))
        
        return critical
    
    def update_inventory(self, resource_type: str, quantity: int):
        """Update inventory for a resource type"""
        self.default_inventory[resource_type] = quantity
        logger.info(f"Inventory updated: {resource_type} = {quantity}")
    
    def get_procurement_plan(self, report: ResourceReport) -> Dict:
        """Generate procurement plan for shortages"""
        plan = {
            "report_id": report.report_id,
            "generated_at": datetime.utcnow().isoformat(),
            "total_procurement_cost": report.total_shortage_cost,
            "items": []
        }
        
        for resource in report.resources:
            if resource.shortage > 0:
                plan["items"].append({
                    "resource": resource.resource_type,
                    "quantity_needed": resource.shortage,
                    "unit": resource.unit,
                    "estimated_cost": resource.shortage * resource.estimated_cost_per_unit,
                    "source": resource.source,
                    "lead_time_minutes": resource.lead_time_minutes,
                    "priority": resource.priority
                })
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        plan["items"].sort(key=lambda x: priority_order.get(x["priority"], 4))
        
        return plan


# Factory function
def create_resource_calculator() -> ResourceCalculator:
    """Create a ResourceCalculator instance"""
    return ResourceCalculator()
