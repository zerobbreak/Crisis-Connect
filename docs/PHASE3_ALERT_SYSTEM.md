# Phase 3: Action-Oriented Alerts & Emergency Integration

## Overview

Phase 3 transforms Phase 2 predictions into **actionable alerts** that drive real emergency response. The key insight: **A prediction nobody acts on is worthless.**

### What Phase 3 Delivers

- **Specific Actions** → What should happen (evacuate, prepare supplies, etc.)
- **Resource Requirements** → How many people, supplies, equipment
- **Timelines** → When to start, when to complete
- **Geographic Specificity** → Exact zones, roads, facilities
- **Multi-Channel Distribution** → SMS, email, sirens, WhatsApp, dashboard
- **Feedback Loop** → Track outcomes, improve predictions

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Phase 2 Output                                │
│                    (AgentOrchestrator.predict())                     │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       AlertFormatter                                 │
│              (CAP-compliant alert formatting)                        │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │ActionGenerator│ │ResourceCalc  │ │AlertDistrib  │
           │(action plans) │ │(resources)   │ │(multi-channel)│
           └──────────────┘ └──────────────┘ └──────────────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  EmergencySystemIntegrator                           │
│         (EOC, hospitals, media, municipal systems)                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FeedbackSystem                                 │
│              (outcome tracking, model improvement)                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Services

### 1. AlertFormatter (`services/alert_formatter.py`)

Converts Phase 2 predictions into CAP (Common Alerting Protocol) compliant alerts.

**Key Features:**
- International CAP standard compliance
- Severity mapping (risk score → EXTREME/SEVERE/MODERATE/MINOR)
- Urgency mapping (hours to peak → IMMEDIATE/EXPECTED/FUTURE)
- Certainty mapping (confidence → OBSERVED/LIKELY/POSSIBLE)
- Automatic instruction generation by risk level
- Support for update and cancel alerts

**Usage:**
```python
from services.alert_formatter import AlertFormatter

formatter = AlertFormatter()
alert = formatter.format_prediction(prediction, location_data)

# Get CAP-compliant dictionary
alert_dict = alert.to_dict()

# Get simplified dictionary for internal use
simple_dict = alert.to_simple_dict()

# Create update or cancel alerts
update = formatter.create_update_alert(original_alert, new_prediction, location_data)
cancel = formatter.create_cancel_alert(original_alert, reason="Conditions improved")
```

### 2. ActionGenerator (`services/action_generator.py`)

Generates specific, actionable emergency response plans.

**Key Features:**
- Hazard-specific actions (flood, drought, storm)
- Priority levels (IMMEDIATE, WITHIN_1HR, WITHIN_3HR, etc.)
- Stakeholder assignment with contact info
- Prerequisites and dependencies
- Success metrics for each action
- Critical path calculation

**Usage:**
```python
from services.action_generator import ActionGenerator

generator = ActionGenerator()
action_plan = generator.generate_actions(prediction, location_data)

# Get actions by priority
immediate = generator.get_immediate_actions(action_plan)

# Get actions for specific stakeholder
fire_actions = generator.get_actions_for_stakeholder(action_plan, "Fire & Rescue")

# Export to dictionary
plan_dict = action_plan.to_dict()
```

**Example Actions for HIGH Flood Risk:**
1. Activate Emergency Operations Center (IMMEDIATE)
2. Issue precautionary evacuation advisory (IMMEDIATE)
3. Open emergency shelters (WITHIN_1HR)
4. Deploy rescue teams (WITHIN_1HR)
5. Alert hospitals (WITHIN_1HR)
6. Mobilize evacuation transport (WITHIN_1HR)

### 3. ResourceCalculator (`services/resource_calculator.py`)

Calculates resources needed based on Sphere humanitarian standards.

**Key Features:**
- Population-based calculations
- Sphere standards compliance (water, shelter, food)
- Gap analysis (needed vs available)
- Cost estimation
- Critical shortage identification
- Procurement plan generation

**Usage:**
```python
from services.resource_calculator import ResourceCalculator

calculator = ResourceCalculator()
report = calculator.calculate_resources(prediction, location_data)

# Get resources by category
by_category = report.get_resources_by_category()

# Generate procurement plan for shortages
procurement = calculator.get_procurement_plan(report)

# Update inventory
calculator.update_inventory("shelter_beds", 3000)
```

**Sphere Standards Applied:**
- Water: 20 liters/person/day
- Shelter: 3.5 m²/person
- Food: 2,100 kcal/person/day
- Latrines: 1 per 20 people
- Health workers: 22 per 10,000 people

### 4. AlertDistributor (`services/alert_distributor.py`)

Multi-channel alert distribution system.

**Channels:**
- SMS - For those without internet
- Email - Detailed information
- Dashboard - Emergency operations centers
- WhatsApp - Mobile users
- Phone Tree (IVR) - Automated calls
- Sirens - Emergency sound alerts
- Telegram - Bot integration

**Usage:**
```python
from services.alert_distributor import AlertDistributor

distributor = AlertDistributor()

# Define recipients
recipients = [
    {"type": "authority", "identifier": "eoc@emergency.gov.za"},
    {"type": "residential", "identifier": "+27123456789"},
    {"type": "critical_facility", "identifier": "hospital@health.gov.za"},
    {"type": "siren_zone", "identifier": "ZONE-JHB-CBD"}
]

# Distribute alert
report = await distributor.distribute_alert(alert, recipients)

# Check results
print(f"Success rate: {report.successful_sends / report.total_recipients:.0%}")
```

**Channel Preferences by Recipient Type:**
- `residential`: SMS, WhatsApp
- `authority`: Email, Dashboard, Phone Tree
- `critical_facility`: Email, Phone Tree, SMS
- `community_leader`: WhatsApp, SMS, Telegram
- `media`: Email, Dashboard
- `siren_zone`: Siren

### 5. EmergencySystemIntegrator (`services/emergency_system_integrator.py`)

Integrates with existing emergency management systems.

**Systems Integrated:**
- Emergency Operations Center (EOC) Dashboard
- Disaster Management Database
- Hospital Network
- Media Broadcast System
- Municipal Alert System

**Usage:**
```python
from services.emergency_system_integrator import EmergencySystemIntegrator

integrator = EmergencySystemIntegrator(config={
    "eoc_api_url": "https://eoc.example.com/api",
    "hospital_api": "https://health.example.com/api"
})

# Integrate alert with all systems
report = await integrator.integrate_alert(
    alert_dict,
    actions=[a.to_dict() for a in action_plan.actions],
    resources=resource_report.to_dict()
)

# Check status
for result in report.results:
    print(f"{result.system_name}: {'OK' if result.success else 'FAILED'}")
```

### 6. FeedbackSystem (`services/feedback_system.py`)

Tracks outcomes and improves predictions over time.

**Key Features:**
- Outcome recording (what actually happened)
- Accuracy analysis (prediction vs actual)
- System-wide metrics (precision, recall, F1)
- Improvement identification
- Trend analysis

**Usage:**
```python
from services.feedback_system import FeedbackSystem, PredictionOutcome

feedback = FeedbackSystem()

# Record what happened
outcome = PredictionOutcome(
    prediction_id="PRED-001",
    alert_id="CrisisConnect-JHB-001",
    location="Johannesburg",
    hazard_type="flood",
    predicted_risk_level="HIGH",
    predicted_risk_score=0.72,
    actual_disaster_occurred=True,
    actual_severity="Severe",
    actions_executed=8,
    actions_planned=10,
    evacuation_rate=0.85,
    false_alarm=False
)

result = await feedback.record_outcome(outcome)

# Get system metrics
metrics = feedback.calculate_system_metrics()
print(f"Accuracy: {metrics.accuracy:.1%}")
print(f"False alarm rate: {metrics.false_alarm_rate:.1%}")

# Get improvement suggestions
improvements = feedback.identify_pattern_improvements()
for imp in improvements:
    print(f"- {imp}")
```

## API Endpoints

### Generate Actionable Alert
```
POST /api/v1/alerts/generate-actionable
```

**Request:**
```json
{
    "location": "Johannesburg",
    "hazard_type": "flood",
    "distribute": false,
    "integrate": false
}
```

**Response:**
```json
{
    "success": true,
    "alert": { /* CAP-compliant alert */ },
    "action_plan": { /* Action plan with stakeholders */ },
    "resources": { /* Resource requirements */ },
    "distribution": { /* If distribute=true */ },
    "integration": { /* If integrate=true */ }
}
```

### Distribute Alert
```
POST /api/v1/alerts/distribute
```

### Record Outcome
```
POST /api/v1/alerts/outcomes
```

### Get System Metrics
```
GET /api/v1/alerts/system-metrics
```

### Get Feedback Report
```
GET /api/v1/alerts/feedback-report?days=30
```

### Preview Alert
```
GET /api/v1/alerts/format-preview/{location}?hazard_type=flood&risk_level=HIGH
```

## Data Models

### FormattedAlert (CAP-compliant)
```python
@dataclass
class FormattedAlert:
    identifier: str       # Unique ID
    sender: str           # "crisis-connect@emergency.gov.za"
    sent: str             # ISO timestamp
    status: AlertStatus   # ACTUAL, EXERCISE, TEST
    msg_type: AlertMessageType  # ALERT, UPDATE, CANCEL
    scope: str            # Public, Restricted, Private
    
    # Content
    headline: str         # "Severe Flood Warning for Johannesburg"
    description: str      # Detailed description
    instruction: str      # What to do
    
    # Classification
    severity: AlertSeverity   # EXTREME, SEVERE, MODERATE, MINOR
    urgency: AlertUrgency     # IMMEDIATE, EXPECTED, FUTURE
    certainty: AlertCertainty # OBSERVED, LIKELY, POSSIBLE
    
    # Timing
    onset: str            # When hazard begins
    expires: str          # When alert expires
    
    # Geography
    areas: List[AlertArea]
```

### Action
```python
@dataclass
class Action:
    action_id: str
    description: str
    category: ActionCategory
    stakeholder: str
    priority: ActionPriority
    prerequisites: List[str]
    estimated_duration_min: int
    success_metric: str
    owner_contact: str
    resources_needed: List[str]
    is_critical: bool
```

### ResourceRequirement
```python
@dataclass
class ResourceRequirement:
    resource_type: str
    category: str
    quantity: int
    unit: str
    estimated_cost_per_unit: float
    total_estimated_cost: float
    availability: int
    shortage: int
    source: str
    lead_time_minutes: int
    priority: str
```

### PredictionOutcome
```python
@dataclass
class PredictionOutcome:
    prediction_id: str
    alert_id: str
    
    # Predicted
    predicted_risk_level: str
    predicted_risk_score: float
    predicted_peak_hours: int
    
    # Actual
    actual_disaster_occurred: bool
    actual_severity: str
    actual_affected_population: int
    
    # Actions
    actions_planned: int
    actions_executed: int
    evacuation_rate: float
    
    # Assessment
    false_alarm: bool
    missed_event: bool
```

## Success Metrics

### Phase 3 Targets
- **Action Execution Rate**: 80%+ of recommended actions completed
- **Evacuation Rate**: 90%+ of at-risk people evacuated
- **Lead Time Used**: Average 24+ hours before peak
- **Distribution Success**: 95%+ message delivery
- **Resource Accuracy**: ±20% of actual needs
- **System Trust**: 80%+ confidence rating from emergency managers

### Tracking
- False alarm rate < 20%
- Miss rate < 10%
- Lives saved (counterfactual estimate)
- Cost-benefit ratio

## Configuration

### Environment Variables
```bash
# SMS Provider
SMS_API_KEY=your_sms_api_key
SMS_SENDER_ID=CRISIS-ALERT

# Email
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=alerts@crisis-connect.gov.za

# Dashboard
DASHBOARD_API_URL=https://eoc.example.com/api
DASHBOARD_API_KEY=your_dashboard_key

# Emergency Systems
EOC_API_URL=https://eoc.gov.za/api
HOSPITAL_API=https://health.gov.za/api
MEDIA_API=https://broadcast.gov.za/api
```

## Testing

Run Phase 3 tests:
```bash
python -m pytest tests/test_phase3_alerts.py -v
```

Run validation:
```bash
python scripts/validate_phase3.py
```

## Example: Full Pipeline

```python
import asyncio
from services.alert_formatter import AlertFormatter
from services.action_generator import ActionGenerator
from services.resource_calculator import ResourceCalculator
from services.alert_distributor import AlertDistributor
from services.emergency_system_integrator import EmergencySystemIntegrator
from services.feedback_system import FeedbackSystem, PredictionOutcome

async def handle_high_risk_prediction(prediction, location_data):
    # 1. Format the alert
    formatter = AlertFormatter()
    alert = formatter.format_prediction(prediction, location_data)
    
    # 2. Generate action plan
    generator = ActionGenerator()
    action_plan = generator.generate_actions(prediction, location_data)
    
    # 3. Calculate resources
    calculator = ResourceCalculator()
    resources = calculator.calculate_resources(prediction, location_data)
    
    # 4. Distribute alert
    distributor = AlertDistributor()
    recipients = get_recipients_for_location(location_data)
    distribution = await distributor.distribute_alert(alert.to_simple_dict(), recipients)
    
    # 5. Integrate with emergency systems
    integrator = EmergencySystemIntegrator()
    integration = await integrator.integrate_alert(
        alert.to_dict(),
        [a.to_dict() for a in action_plan.actions],
        resources.to_dict()
    )
    
    return {
        "alert_id": alert.identifier,
        "actions": len(action_plan.actions),
        "resources_needed": len(resources.resources),
        "distribution_success": distribution.successful_sends / distribution.total_recipients,
        "systems_integrated": integration.successful
    }

# After the event, record outcome
async def record_event_outcome(alert_id, actual_data):
    feedback = FeedbackSystem()
    outcome = PredictionOutcome(
        prediction_id=alert_id,
        alert_id=alert_id,
        actual_disaster_occurred=actual_data["occurred"],
        # ... other fields
    )
    return await feedback.record_outcome(outcome)
```

## Troubleshooting

### Alert Not Distributing
1. Check channel configuration (API keys)
2. Verify recipient format (phone numbers, emails)
3. Check rate limits on external services
4. Review logs for specific errors

### Integration Failures
1. Verify external system endpoints are accessible
2. Check authentication credentials
3. Review payload format requirements
4. Check network connectivity

### Feedback Not Recording
1. Ensure data directory is writable
2. Check prediction_id matches existing prediction
3. Verify outcome data completeness

## Future Enhancements (Phase 4+)

- Real-time integration with actual emergency systems
- Live pilot in selected regions
- Emergency manager training module
- Integration with climate change projections
- Automatic model retraining based on feedback
- Multi-language alert generation
- Accessibility features (audio alerts, simple language)
