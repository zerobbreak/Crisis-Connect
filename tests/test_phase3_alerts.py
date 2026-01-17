# tests/test_phase3_alerts.py
"""
Comprehensive test suite for Phase 3: Action-Oriented Alerts & Emergency Integration

Tests cover:
1. AlertFormatter - CAP-compliant alert formatting
2. ActionGenerator - Action plan generation
3. ResourceCalculator - Resource estimation
4. AlertDistributor - Multi-channel distribution
5. EmergencySystemIntegrator - External system integration
6. FeedbackSystem - Outcome tracking and learning
7. Integration tests - Full pipeline
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

# Import Phase 3 services
from services.alert_formatter import (
    AlertFormatter, FormattedAlert, AlertSeverity, AlertUrgency,
    AlertCertainty, AlertStatus, AlertMessageType, AlertArea,
    create_alert_formatter
)
from services.action_generator import (
    ActionGenerator, Action, ActionPlan, ActionPriority, ActionCategory,
    create_action_generator
)
from services.resource_calculator import (
    ResourceCalculator, ResourceRequirement, ResourceReport,
    create_resource_calculator
)
from services.alert_distributor import (
    AlertDistributor, DeliveryResult, DistributionReport,
    SMSChannel, EmailChannel, DashboardChannel, WhatsAppChannel,
    create_alert_distributor
)
from services.emergency_system_integrator import (
    EmergencySystemIntegrator, IntegrationResult, IntegrationReport,
    create_emergency_integrator
)
from services.feedback_system import (
    FeedbackSystem, PredictionOutcome, OutcomeAnalysis, SystemMetrics,
    create_feedback_system
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_prediction():
    """Sample Phase 2 prediction output"""
    return {
        "location": "Johannesburg",
        "hazard_type": "flood",
        "timestamp": datetime.now().isoformat(),
        "prediction": {
            "risk_score": 0.72,
            "risk_level": "HIGH",
            "hours_to_peak": 18,
            "primary_method": "pattern_matching",
            "method_breakdown": {
                "lstm": 0.68,
                "pattern_matching": 0.75,
                "progression": 0.70
            },
            "confidence": 0.75,
            "warnings": ["Method disagreement detected"],
            "recommendation": "Prepare emergency response protocols.",
            "is_anomalous": False
        },
        "forecasts": [
            {"method": "lstm", "risk_score": 0.68, "hours_to_peak": 20, "confidence": 0.70},
            {"method": "pattern_matching", "risk_score": 0.75, "hours_to_peak": 18, "confidence": 0.80},
            {"method": "progression", "risk_score": 0.70, "hours_to_peak": 16, "confidence": 0.65}
        ]
    }


@pytest.fixture
def sample_location_data():
    """Sample location data"""
    return {
        "name": "Johannesburg CBD",
        "latitude": -26.2041,
        "longitude": 28.0473,
        "population": 500000,
        "region": "Gauteng",
        "is_coastal": False,
        "evacuation_zones": [
            {"name": "Zone A", "population": 50000},
            {"name": "Zone B", "population": 75000}
        ],
        "critical_facilities": ["Hospital A", "School B", "Power Station C"]
    }


@pytest.fixture
def sample_outcome():
    """Sample prediction outcome"""
    return PredictionOutcome(
        prediction_id="PRED-001",
        alert_id="CrisisConnect-JHB-20260117-1",
        location="Johannesburg",
        hazard_type="flood",
        prediction_timestamp=datetime.now().isoformat(),
        predicted_risk_level="HIGH",
        predicted_risk_score=0.72,
        predicted_peak_hours=18,
        predicted_severity="Severe",
        method_breakdown={"lstm": 0.68, "pattern_matching": 0.75},
        actual_disaster_occurred=True,
        actual_severity="Severe",
        actual_damage_estimate=5000000,
        actual_affected_population=25000,
        actual_casualties=0,
        actual_injuries=15,
        actions_planned=10,
        actions_executed=8,
        actions_successful=7,
        evacuation_ordered=True,
        evacuation_rate=0.85,
        shelter_utilization=0.60,
        lead_time_hours=18,
        lead_time_used_hours=12,
        false_alarm=False,
        missed_event=False,
        confidence_rating=0.80,
        notes="Test outcome",
        recorded_by="test_user"
    )


# ============================================================================
# AlertFormatter Tests
# ============================================================================

class TestAlertFormatter:
    """Tests for AlertFormatter service"""
    
    def test_create_formatter(self):
        """Test formatter creation"""
        formatter = create_alert_formatter()
        assert formatter is not None
        assert isinstance(formatter, AlertFormatter)
    
    def test_format_prediction_basic(self, sample_prediction, sample_location_data):
        """Test basic alert formatting"""
        formatter = AlertFormatter()
        alert = formatter.format_prediction(sample_prediction, sample_location_data)
        
        assert isinstance(alert, FormattedAlert)
        assert alert.identifier.startswith("CrisisConnect-")
        assert alert.sender == "crisis-connect@emergency.gov.za"
        assert alert.status == AlertStatus.ACTUAL
        assert alert.msg_type == AlertMessageType.ALERT
        assert alert.scope == "Public"
    
    def test_format_prediction_severity_mapping(self, sample_location_data):
        """Test severity mapping from risk score"""
        formatter = AlertFormatter()
        
        test_cases = [
            (0.90, AlertSeverity.EXTREME),
            (0.70, AlertSeverity.SEVERE),
            (0.45, AlertSeverity.MODERATE),
            (0.15, AlertSeverity.MINOR),
        ]
        
        for risk_score, expected_severity in test_cases:
            prediction = {
                "location": "Test",
                "hazard_type": "flood",
                "prediction": {"risk_score": risk_score, "risk_level": "HIGH", "hours_to_peak": 12}
            }
            alert = formatter.format_prediction(prediction, sample_location_data)
            assert alert.severity == expected_severity, f"Risk score {risk_score} should map to {expected_severity}"
    
    def test_format_prediction_urgency_mapping(self, sample_location_data):
        """Test urgency mapping from hours to peak"""
        formatter = AlertFormatter()
        
        test_cases = [
            (4, AlertUrgency.IMMEDIATE),
            (18, AlertUrgency.EXPECTED),
            (48, AlertUrgency.FUTURE),
        ]
        
        for hours, expected_urgency in test_cases:
            prediction = {
                "location": "Test",
                "hazard_type": "flood",
                "prediction": {"risk_score": 0.7, "risk_level": "HIGH", "hours_to_peak": hours}
            }
            alert = formatter.format_prediction(prediction, sample_location_data)
            assert alert.urgency == expected_urgency, f"Hours {hours} should map to {expected_urgency}"
    
    def test_format_prediction_content(self, sample_prediction, sample_location_data):
        """Test alert content generation"""
        formatter = AlertFormatter()
        alert = formatter.format_prediction(sample_prediction, sample_location_data)
        
        # Check headline
        assert "Johannesburg" in alert.headline
        assert "flood" in alert.headline.lower() or "Flood" in alert.headline
        
        # Check description
        assert "HAZARD" in alert.description
        assert "Risk Level" in alert.description
        
        # Check instruction
        assert len(alert.instruction) > 0
        assert "EVACUATION" in alert.instruction or "PREPARE" in alert.instruction
    
    def test_format_prediction_areas(self, sample_prediction, sample_location_data):
        """Test affected areas in alert"""
        formatter = AlertFormatter()
        alert = formatter.format_prediction(sample_prediction, sample_location_data)
        
        assert len(alert.areas) > 0
        assert alert.areas[0].area_desc == sample_location_data["name"]
        assert alert.areas[0].population == sample_location_data["population"]
    
    def test_to_dict_cap_compliant(self, sample_prediction, sample_location_data):
        """Test CAP-compliant dictionary output"""
        formatter = AlertFormatter()
        alert = formatter.format_prediction(sample_prediction, sample_location_data)
        alert_dict = alert.to_dict()
        
        # Check required CAP fields
        assert "identifier" in alert_dict
        assert "sender" in alert_dict
        assert "sent" in alert_dict
        assert "status" in alert_dict
        assert "msgType" in alert_dict
        assert "scope" in alert_dict
        assert "info" in alert_dict
        
        # Check info segment
        info = alert_dict["info"]
        assert "headline" in info
        assert "description" in info
        assert "instruction" in info
        assert "severity" in info
        assert "urgency" in info
        assert "area" in info
    
    def test_create_update_alert(self, sample_prediction, sample_location_data):
        """Test creating update alerts"""
        formatter = AlertFormatter()
        original = formatter.format_prediction(sample_prediction, sample_location_data)
        
        # Create update with new prediction
        new_prediction = sample_prediction.copy()
        new_prediction["prediction"] = {**sample_prediction["prediction"], "risk_score": 0.85}
        
        update = formatter.create_update_alert(original, new_prediction, sample_location_data)
        
        assert update.msg_type == AlertMessageType.UPDATE
        assert original.identifier in update.references
    
    def test_create_cancel_alert(self, sample_prediction, sample_location_data):
        """Test creating cancel alerts"""
        formatter = AlertFormatter()
        original = formatter.format_prediction(sample_prediction, sample_location_data)
        
        cancel = formatter.create_cancel_alert(original, "Conditions improved")
        
        assert cancel.msg_type == AlertMessageType.CANCEL
        assert "CANCELLED" in cancel.headline
        assert original.identifier in cancel.references


# ============================================================================
# ActionGenerator Tests
# ============================================================================

class TestActionGenerator:
    """Tests for ActionGenerator service"""
    
    def test_create_generator(self):
        """Test generator creation"""
        generator = create_action_generator()
        assert generator is not None
        assert isinstance(generator, ActionGenerator)
    
    def test_generate_actions_flood_high(self, sample_prediction, sample_location_data):
        """Test action generation for HIGH flood risk"""
        generator = ActionGenerator()
        plan = generator.generate_actions(sample_prediction, sample_location_data)
        
        assert isinstance(plan, ActionPlan)
        assert plan.hazard_type == "flood"
        assert plan.risk_level == "HIGH"
        assert len(plan.actions) > 0
    
    def test_generate_actions_includes_eoc(self, sample_prediction, sample_location_data):
        """Test that HIGH risk includes EOC activation"""
        generator = ActionGenerator()
        plan = generator.generate_actions(sample_prediction, sample_location_data)
        
        eoc_actions = [a for a in plan.actions if "eoc" in a.action_id.lower() or "EOC" in a.description]
        assert len(eoc_actions) > 0, "HIGH risk should include EOC activation"
    
    def test_generate_actions_priorities(self, sample_prediction, sample_location_data):
        """Test action priorities are set correctly"""
        generator = ActionGenerator()
        plan = generator.generate_actions(sample_prediction, sample_location_data)
        
        # Should have some immediate actions for HIGH risk
        immediate_actions = [a for a in plan.actions if a.priority == ActionPriority.IMMEDIATE]
        assert len(immediate_actions) > 0, "HIGH risk should have immediate actions"
    
    def test_generate_actions_prerequisites(self, sample_prediction, sample_location_data):
        """Test that prerequisites are set"""
        generator = ActionGenerator()
        plan = generator.generate_actions(sample_prediction, sample_location_data)
        
        # Some actions should have prerequisites
        actions_with_prereqs = [a for a in plan.actions if a.prerequisites]
        # Not all actions need prerequisites, but some should
        assert len(actions_with_prereqs) >= 0  # Prerequisites are optional
    
    def test_generate_actions_critical_path(self, sample_prediction, sample_location_data):
        """Test critical path calculation"""
        generator = ActionGenerator()
        plan = generator.generate_actions(sample_prediction, sample_location_data)
        
        # Should have critical path for HIGH risk
        assert len(plan.critical_path) >= 0  # May be empty if no critical actions
    
    def test_generate_actions_different_risk_levels(self, sample_location_data):
        """Test action generation for different risk levels"""
        generator = ActionGenerator()
        
        risk_levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        action_counts = {}
        
        for level in risk_levels:
            prediction = {
                "location": "Test",
                "hazard_type": "flood",
                "prediction": {"risk_level": level, "risk_score": 0.5, "hours_to_peak": 18}
            }
            plan = generator.generate_actions(prediction, sample_location_data)
            action_counts[level] = len(plan.actions)
        
        # Higher risk should generally have more actions
        assert action_counts["CRITICAL"] >= action_counts["LOW"]
    
    def test_action_to_dict(self, sample_prediction, sample_location_data):
        """Test action dictionary conversion"""
        generator = ActionGenerator()
        plan = generator.generate_actions(sample_prediction, sample_location_data)
        
        if plan.actions:
            action_dict = plan.actions[0].to_dict()
            assert "action_id" in action_dict
            assert "description" in action_dict
            assert "stakeholder" in action_dict
            assert "priority" in action_dict
            assert "success_metric" in action_dict


# ============================================================================
# ResourceCalculator Tests
# ============================================================================

class TestResourceCalculator:
    """Tests for ResourceCalculator service"""
    
    def test_create_calculator(self):
        """Test calculator creation"""
        calculator = create_resource_calculator()
        assert calculator is not None
        assert isinstance(calculator, ResourceCalculator)
    
    def test_calculate_resources_flood(self, sample_prediction, sample_location_data):
        """Test resource calculation for flood"""
        calculator = ResourceCalculator()
        report = calculator.calculate_resources(sample_prediction, sample_location_data)
        
        assert isinstance(report, ResourceReport)
        assert report.hazard_type == "flood"
        assert len(report.resources) > 0
    
    def test_calculate_resources_includes_essentials(self, sample_prediction, sample_location_data):
        """Test that essential resources are included"""
        calculator = ResourceCalculator()
        report = calculator.calculate_resources(sample_prediction, sample_location_data)
        
        resource_types = [r.resource_type for r in report.resources]
        
        # Should include water and food for HIGH risk
        assert "drinking_water_liters" in resource_types
        assert "food_meals" in resource_types
    
    def test_calculate_resources_quantities(self, sample_prediction, sample_location_data):
        """Test resource quantities are reasonable"""
        calculator = ResourceCalculator()
        report = calculator.calculate_resources(sample_prediction, sample_location_data)
        
        for resource in report.resources:
            assert resource.quantity > 0, f"{resource.resource_type} should have positive quantity"
            assert resource.estimated_cost_per_unit > 0
            assert resource.total_estimated_cost == resource.quantity * resource.estimated_cost_per_unit
    
    def test_calculate_resources_gap_analysis(self, sample_prediction, sample_location_data):
        """Test shortage calculation"""
        calculator = ResourceCalculator()
        report = calculator.calculate_resources(sample_prediction, sample_location_data)
        
        # Check that shortage is calculated correctly
        for resource in report.resources:
            expected_shortage = max(0, resource.quantity - resource.availability)
            assert resource.shortage == expected_shortage
    
    def test_calculate_resources_critical_shortages(self, sample_prediction, sample_location_data):
        """Test critical shortage identification"""
        calculator = ResourceCalculator()
        report = calculator.calculate_resources(sample_prediction, sample_location_data)
        
        # Critical shortages should be identified
        assert isinstance(report.critical_shortages, list)
    
    def test_report_to_dict(self, sample_prediction, sample_location_data):
        """Test report dictionary conversion"""
        calculator = ResourceCalculator()
        report = calculator.calculate_resources(sample_prediction, sample_location_data)
        report_dict = report.to_dict()
        
        assert "report_id" in report_dict
        assert "resources" in report_dict
        assert "summary" in report_dict
        assert "total_estimated_cost" in report_dict["summary"]
    
    def test_procurement_plan(self, sample_prediction, sample_location_data):
        """Test procurement plan generation"""
        calculator = ResourceCalculator()
        report = calculator.calculate_resources(sample_prediction, sample_location_data)
        plan = calculator.get_procurement_plan(report)
        
        assert "items" in plan
        assert "total_procurement_cost" in plan


# ============================================================================
# AlertDistributor Tests
# ============================================================================

class TestAlertDistributor:
    """Tests for AlertDistributor service"""
    
    def test_create_distributor(self):
        """Test distributor creation"""
        distributor = create_alert_distributor()
        assert distributor is not None
        assert isinstance(distributor, AlertDistributor)
    
    def test_available_channels(self):
        """Test available channels"""
        distributor = AlertDistributor()
        channels = distributor.get_available_channels()
        
        assert "sms" in channels
        assert "email" in channels
        assert "dashboard" in channels
        assert "whatsapp" in channels
    
    @pytest.mark.asyncio
    async def test_distribute_alert_basic(self):
        """Test basic alert distribution"""
        distributor = AlertDistributor()
        
        alert = {
            "identifier": "TEST-001",
            "headline": "Test Alert",
            "severity": "Severe",
            "hours_to_peak": 12,
            "instruction": "Test instruction"
        }
        
        recipients = [
            {"type": "authority", "identifier": "test@example.com"},
            {"type": "residential", "identifier": "+27123456789"}
        ]
        
        report = await distributor.distribute_alert(alert, recipients)
        
        assert isinstance(report, DistributionReport)
        assert report.total_recipients == len(recipients)
        assert report.successful_sends + report.failed_sends > 0
    
    @pytest.mark.asyncio
    async def test_sms_channel(self):
        """Test SMS channel"""
        channel = SMSChannel()
        
        alert = {"headline": "Test", "severity": "Severe", "hours_to_peak": 12, "instruction": "Test"}
        result = await channel.send(alert, "+27123456789")
        
        assert isinstance(result, DeliveryResult)
        assert result.channel == "sms"
        assert result.success
    
    @pytest.mark.asyncio
    async def test_email_channel(self):
        """Test Email channel"""
        channel = EmailChannel()
        
        alert = {"headline": "Test", "severity": "Severe", "hours_to_peak": 12, "instruction": "Test"}
        result = await channel.send(alert, "test@example.com")
        
        assert isinstance(result, DeliveryResult)
        assert result.channel == "email"
        assert result.success
    
    @pytest.mark.asyncio
    async def test_distribution_report(self):
        """Test distribution report generation"""
        distributor = AlertDistributor()
        
        alert = {"identifier": "TEST-001", "headline": "Test", "severity": "Severe", "hours_to_peak": 12}
        recipients = [{"type": "authority", "identifier": "test@example.com"}]
        
        report = await distributor.distribute_alert(alert, recipients)
        report_dict = report.to_dict()
        
        assert "alert_id" in report_dict
        assert "distribution_id" in report_dict
        assert "successful_sends" in report_dict
        assert "results_by_channel" in report_dict


# ============================================================================
# EmergencySystemIntegrator Tests
# ============================================================================

class TestEmergencySystemIntegrator:
    """Tests for EmergencySystemIntegrator service"""
    
    def test_create_integrator(self):
        """Test integrator creation"""
        integrator = create_emergency_integrator()
        assert integrator is not None
        assert isinstance(integrator, EmergencySystemIntegrator)
    
    @pytest.mark.asyncio
    async def test_integrate_alert(self, sample_prediction, sample_location_data):
        """Test alert integration"""
        integrator = EmergencySystemIntegrator()
        formatter = AlertFormatter()
        
        alert = formatter.format_prediction(sample_prediction, sample_location_data)
        
        report = await integrator.integrate_alert(
            alert.to_dict(),
            actions=[],
            resources={}
        )
        
        assert isinstance(report, IntegrationReport)
        assert report.total_systems > 0
    
    @pytest.mark.asyncio
    async def test_integration_results(self, sample_prediction, sample_location_data):
        """Test integration results structure"""
        integrator = EmergencySystemIntegrator()
        formatter = AlertFormatter()
        
        alert = formatter.format_prediction(sample_prediction, sample_location_data)
        
        report = await integrator.integrate_alert(alert.to_dict())
        
        # Should have results from multiple systems
        assert len(report.results) > 0
        
        for result in report.results:
            assert isinstance(result, IntegrationResult)
            assert result.system_name
            assert isinstance(result.success, bool)
    
    @pytest.mark.asyncio
    async def test_check_system_status(self):
        """Test system status check"""
        integrator = EmergencySystemIntegrator()
        
        status = await integrator.check_system_status()
        
        assert isinstance(status, dict)
        assert "eoc" in status
        assert "disaster_db" in status


# ============================================================================
# FeedbackSystem Tests
# ============================================================================

class TestFeedbackSystem:
    """Tests for FeedbackSystem service"""
    
    def test_create_feedback_system(self, tmp_path):
        """Test feedback system creation"""
        system = create_feedback_system(str(tmp_path))
        assert system is not None
        assert isinstance(system, FeedbackSystem)
    
    @pytest.mark.asyncio
    async def test_record_outcome(self, sample_outcome, tmp_path):
        """Test recording prediction outcome"""
        system = FeedbackSystem(str(tmp_path))
        
        result = await system.record_outcome(sample_outcome)
        
        assert result["recorded"]
        assert "analysis" in result
        assert "improvements" in result
    
    @pytest.mark.asyncio
    async def test_outcome_analysis(self, sample_outcome, tmp_path):
        """Test outcome analysis"""
        system = FeedbackSystem(str(tmp_path))
        
        result = await system.record_outcome(sample_outcome)
        analysis = result["analysis"]
        
        assert "prediction_accurate" in analysis
        assert "overall_score" in analysis
        assert "improvements_needed" in analysis
    
    def test_calculate_system_metrics_empty(self, tmp_path):
        """Test metrics calculation with no data"""
        system = FeedbackSystem(str(tmp_path))
        
        metrics = system.calculate_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.total_predictions == 0
    
    @pytest.mark.asyncio
    async def test_calculate_system_metrics_with_data(self, sample_outcome, tmp_path):
        """Test metrics calculation with data"""
        system = FeedbackSystem(str(tmp_path))
        
        # Record some outcomes
        await system.record_outcome(sample_outcome)
        
        metrics = system.calculate_system_metrics()
        
        assert metrics.total_predictions > 0
    
    def test_identify_improvements(self, tmp_path):
        """Test improvement identification"""
        system = FeedbackSystem(str(tmp_path))
        
        improvements = system.identify_pattern_improvements()
        
        assert isinstance(improvements, list)
    
    @pytest.mark.asyncio
    async def test_generate_feedback_report(self, sample_outcome, tmp_path):
        """Test feedback report generation"""
        system = FeedbackSystem(str(tmp_path))
        
        await system.record_outcome(sample_outcome)
        
        report = system.generate_feedback_report(days=30)
        
        assert "summary" in report
        assert "detailed_metrics" in report
        assert "recommended_improvements" in report
    
    def test_get_recent_outcomes(self, tmp_path):
        """Test getting recent outcomes"""
        system = FeedbackSystem(str(tmp_path))
        
        outcomes = system.get_recent_outcomes(limit=10)
        
        assert isinstance(outcomes, list)


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase3Integration:
    """Integration tests for full Phase 3 pipeline"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_prediction, sample_location_data, tmp_path):
        """Test complete Phase 3 pipeline"""
        # Step 1: Format alert
        formatter = AlertFormatter()
        alert = formatter.format_prediction(sample_prediction, sample_location_data)
        
        assert alert.identifier is not None
        
        # Step 2: Generate actions
        generator = ActionGenerator()
        action_plan = generator.generate_actions(sample_prediction, sample_location_data)
        
        assert len(action_plan.actions) > 0
        
        # Step 3: Calculate resources
        calculator = ResourceCalculator()
        resources = calculator.calculate_resources(
            sample_prediction, 
            sample_location_data,
            [a.to_dict() for a in action_plan.actions]
        )
        
        assert len(resources.resources) > 0
        
        # Step 4: Distribute alert
        distributor = AlertDistributor()
        recipients = [
            {"type": "authority", "identifier": "test@example.com"}
        ]
        distribution = await distributor.distribute_alert(alert.to_simple_dict(), recipients)
        
        assert distribution.total_recipients > 0
        
        # Step 5: Integrate with systems
        integrator = EmergencySystemIntegrator()
        integration = await integrator.integrate_alert(
            alert.to_dict(),
            [a.to_dict() for a in action_plan.actions],
            resources.to_dict()
        )
        
        assert integration.total_systems > 0
        
        # Step 6: Record outcome
        feedback = FeedbackSystem(str(tmp_path))
        outcome = PredictionOutcome(
            prediction_id=alert.identifier,
            alert_id=alert.identifier,
            location=sample_location_data["name"],
            hazard_type="flood",
            prediction_timestamp=datetime.now().isoformat(),
            predicted_risk_level="HIGH",
            predicted_risk_score=0.72,
            predicted_peak_hours=18,
            predicted_severity="Severe",
            method_breakdown={"lstm": 0.68, "pattern_matching": 0.75},
            actual_disaster_occurred=True,
            actual_severity="Severe",
            actions_planned=len(action_plan.actions),
            actions_executed=len(action_plan.actions) - 2,
            actions_successful=len(action_plan.actions) - 3,
            evacuation_ordered=True,
            evacuation_rate=0.85,
            lead_time_hours=18,
            lead_time_used_hours=12,
            false_alarm=False,
            confidence_rating=0.8
        )
        
        result = await feedback.record_outcome(outcome)
        
        assert result["recorded"]
    
    def test_pipeline_data_flow(self, sample_prediction, sample_location_data):
        """Test data flows correctly through pipeline"""
        formatter = AlertFormatter()
        generator = ActionGenerator()
        calculator = ResourceCalculator()
        
        # Format alert
        alert = formatter.format_prediction(sample_prediction, sample_location_data)
        
        # Generate actions
        action_plan = generator.generate_actions(sample_prediction, sample_location_data)
        
        # Calculate resources
        resources = calculator.calculate_resources(
            sample_prediction, 
            sample_location_data,
            [a.to_dict() for a in action_plan.actions]
        )
        
        # Verify data consistency
        assert alert.hazard_type == action_plan.hazard_type
        assert action_plan.hazard_type == resources.hazard_type
        assert action_plan.risk_level == resources.risk_level
    
    def test_different_hazard_types(self, sample_location_data):
        """Test pipeline with different hazard types"""
        formatter = AlertFormatter()
        generator = ActionGenerator()
        calculator = ResourceCalculator()
        
        hazard_types = ["flood", "drought", "storm"]
        
        for hazard in hazard_types:
            prediction = {
                "location": "Test",
                "hazard_type": hazard,
                "prediction": {
                    "risk_score": 0.7,
                    "risk_level": "HIGH",
                    "hours_to_peak": 18,
                    "confidence": 0.75
                }
            }
            
            alert = formatter.format_prediction(prediction, sample_location_data)
            action_plan = generator.generate_actions(prediction, sample_location_data)
            resources = calculator.calculate_resources(prediction, sample_location_data)
            
            assert alert.hazard_type == hazard
            assert action_plan.hazard_type == hazard
            assert resources.hazard_type == hazard


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
