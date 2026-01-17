#!/usr/bin/env python
"""
Phase 3 Validation Script

Validates the Phase 3 implementation by:
1. Checking all services can be imported and instantiated
2. Testing core functionality of each service
3. Running the full alert pipeline
4. Verifying API endpoints are registered
5. Checking documentation and test status
"""

import sys
import asyncio
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_check(name: str, passed: bool, details: str = ""):
    """Print check result"""
    status = "PASS" if passed else "FAIL"
    marker = "[+]" if passed else "[-]"
    print(f"  {marker} {status}: {name}")
    if details:
        print(f"          {details}")


def main():
    """Run all Phase 3 validation checks"""
    print("\n" + "="*60)
    print("  PHASE 3 VALIDATION: Action-Oriented Alerts")
    print("="*60)
    print(f"  Timestamp: {datetime.now().isoformat()}")
    
    results = []
    
    # =========================================================================
    # Check 1: Service Imports
    # =========================================================================
    print_header("1. Service Imports")
    
    services_to_check = [
        ("AlertFormatter", "services.alert_formatter", "AlertFormatter"),
        ("ActionGenerator", "services.action_generator", "ActionGenerator"),
        ("ResourceCalculator", "services.resource_calculator", "ResourceCalculator"),
        ("AlertDistributor", "services.alert_distributor", "AlertDistributor"),
        ("EmergencySystemIntegrator", "services.emergency_system_integrator", "EmergencySystemIntegrator"),
        ("FeedbackSystem", "services.feedback_system", "FeedbackSystem"),
    ]
    
    for name, module, class_name in services_to_check:
        try:
            mod = __import__(module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            instance = cls() if class_name != "FeedbackSystem" else cls(str(project_root / "data"))
            print_check(f"Import {name}", True)
            results.append(True)
        except Exception as e:
            print_check(f"Import {name}", False, str(e))
            results.append(False)
    
    # =========================================================================
    # Check 2: Alert Formatter
    # =========================================================================
    print_header("2. Alert Formatter")
    
    try:
        from services.alert_formatter import AlertFormatter, AlertSeverity, AlertUrgency
        
        formatter = AlertFormatter()
        
        # Test prediction formatting
        prediction = {
            "location": "Test Location",
            "hazard_type": "flood",
            "prediction": {
                "risk_score": 0.72,
                "risk_level": "HIGH",
                "hours_to_peak": 18,
                "confidence": 0.75,
                "method_breakdown": {"lstm": 0.7, "pattern": 0.75}
            }
        }
        
        location_data = {
            "name": "Test Location",
            "latitude": -26.2,
            "longitude": 28.0,
            "population": 100000,
            "region": "Test Region"
        }
        
        alert = formatter.format_prediction(prediction, location_data)
        
        # Verify CAP compliance
        alert_dict = alert.to_dict()
        cap_fields = ["identifier", "sender", "sent", "status", "msgType", "scope", "info"]
        missing_fields = [f for f in cap_fields if f not in alert_dict]
        
        if not missing_fields:
            print_check("CAP-compliant format", True, f"All required fields present")
            results.append(True)
        else:
            print_check("CAP-compliant format", False, f"Missing: {missing_fields}")
            results.append(False)
        
        # Check severity mapping
        if alert.severity == AlertSeverity.SEVERE:
            print_check("Severity mapping", True, f"0.72 -> SEVERE")
            results.append(True)
        else:
            print_check("Severity mapping", False, f"Expected SEVERE, got {alert.severity}")
            results.append(False)
        
        # Check urgency mapping
        if alert.urgency == AlertUrgency.EXPECTED:
            print_check("Urgency mapping", True, f"18h -> EXPECTED")
            results.append(True)
        else:
            print_check("Urgency mapping", False, f"Expected EXPECTED, got {alert.urgency}")
            results.append(False)
            
    except Exception as e:
        print_check("Alert Formatter", False, str(e))
        results.extend([False, False, False])
    
    # =========================================================================
    # Check 3: Action Generator
    # =========================================================================
    print_header("3. Action Generator")
    
    try:
        from services.action_generator import ActionGenerator, ActionPriority
        
        generator = ActionGenerator()
        plan = generator.generate_actions(prediction, location_data)
        
        # Check actions generated
        if len(plan.actions) > 0:
            print_check("Actions generated", True, f"{len(plan.actions)} actions")
            results.append(True)
        else:
            print_check("Actions generated", False, "No actions")
            results.append(False)
        
        # Check immediate actions for HIGH risk
        immediate = [a for a in plan.actions if a.priority == ActionPriority.IMMEDIATE]
        if len(immediate) > 0:
            print_check("Immediate actions", True, f"{len(immediate)} immediate actions")
            results.append(True)
        else:
            print_check("Immediate actions", False, "No immediate actions for HIGH risk")
            results.append(False)
        
        # Check action structure
        if plan.actions[0].stakeholder and plan.actions[0].success_metric:
            print_check("Action structure", True, "Stakeholder and success metric present")
            results.append(True)
        else:
            print_check("Action structure", False, "Missing required fields")
            results.append(False)
            
    except Exception as e:
        print_check("Action Generator", False, str(e))
        results.extend([False, False, False])
    
    # =========================================================================
    # Check 4: Resource Calculator
    # =========================================================================
    print_header("4. Resource Calculator")
    
    try:
        from services.resource_calculator import ResourceCalculator
        
        calculator = ResourceCalculator()
        report = calculator.calculate_resources(prediction, location_data)
        
        # Check resources calculated
        if len(report.resources) > 0:
            print_check("Resources calculated", True, f"{len(report.resources)} resource types")
            results.append(True)
        else:
            print_check("Resources calculated", False, "No resources")
            results.append(False)
        
        # Check essential resources
        resource_types = [r.resource_type for r in report.resources]
        essentials = ["drinking_water_liters", "food_meals"]
        missing = [e for e in essentials if e not in resource_types]
        
        if not missing:
            print_check("Essential resources", True, "Water and food included")
            results.append(True)
        else:
            print_check("Essential resources", False, f"Missing: {missing}")
            results.append(False)
        
        # Check cost calculation
        if report.total_estimated_cost > 0:
            print_check("Cost calculation", True, f"R{report.total_estimated_cost:,.2f}")
            results.append(True)
        else:
            print_check("Cost calculation", False, "No cost calculated")
            results.append(False)
            
    except Exception as e:
        print_check("Resource Calculator", False, str(e))
        results.extend([False, False, False])
    
    # =========================================================================
    # Check 5: Alert Distributor
    # =========================================================================
    print_header("5. Alert Distributor")
    
    try:
        from services.alert_distributor import AlertDistributor
        
        distributor = AlertDistributor()
        
        # Check channels
        channels = distributor.get_available_channels()
        expected_channels = ["sms", "email", "dashboard", "whatsapp"]
        missing_channels = [c for c in expected_channels if c not in channels]
        
        if not missing_channels:
            print_check("Distribution channels", True, f"{len(channels)} channels available")
            results.append(True)
        else:
            print_check("Distribution channels", False, f"Missing: {missing_channels}")
            results.append(False)
        
        # Test distribution (async)
        async def test_distribution():
            alert_simple = {
                "identifier": "TEST-001",
                "headline": "Test Alert",
                "severity": "Severe",
                "hours_to_peak": 12,
                "instruction": "Test"
            }
            recipients = [{"type": "authority", "identifier": "test@example.com"}]
            report = await distributor.distribute_alert(alert_simple, recipients)
            return report
        
        dist_report = asyncio.run(test_distribution())
        
        if dist_report.total_recipients > 0:
            print_check("Alert distribution", True, f"{dist_report.successful_sends} successful")
            results.append(True)
        else:
            print_check("Alert distribution", False, "No recipients processed")
            results.append(False)
            
    except Exception as e:
        print_check("Alert Distributor", False, str(e))
        results.extend([False, False])
    
    # =========================================================================
    # Check 6: Emergency System Integrator
    # =========================================================================
    print_header("6. Emergency System Integrator")
    
    try:
        from services.emergency_system_integrator import EmergencySystemIntegrator
        
        integrator = EmergencySystemIntegrator()
        
        # Test integration (async)
        async def test_integration():
            return await integrator.integrate_alert(alert.to_dict())
        
        int_report = asyncio.run(test_integration())
        
        if int_report.total_systems > 0:
            print_check("System integration", True, f"{int_report.successful}/{int_report.total_systems} systems")
            results.append(True)
        else:
            print_check("System integration", False, "No systems integrated")
            results.append(False)
        
        # Check system types
        system_names = [r.system_name for r in int_report.results]
        if "EOC Dashboard" in system_names:
            print_check("EOC integration", True, "EOC Dashboard connected")
            results.append(True)
        else:
            print_check("EOC integration", False, "EOC not in results")
            results.append(False)
            
    except Exception as e:
        print_check("Emergency Integrator", False, str(e))
        results.extend([False, False])
    
    # =========================================================================
    # Check 7: Feedback System
    # =========================================================================
    print_header("7. Feedback System")
    
    try:
        from services.feedback_system import FeedbackSystem, PredictionOutcome
        
        feedback = FeedbackSystem(str(project_root / "data"))
        
        # Test outcome recording
        outcome = PredictionOutcome(
            prediction_id="VAL-001",
            alert_id="CrisisConnect-VAL-001",
            location="Validation Test",
            hazard_type="flood",
            prediction_timestamp=datetime.now().isoformat(),
            predicted_risk_level="HIGH",
            predicted_risk_score=0.72,
            predicted_peak_hours=18,
            predicted_severity="Severe",
            method_breakdown={"lstm": 0.7},
            actual_disaster_occurred=True,
            actual_severity="Severe",
            actions_planned=10,
            actions_executed=8,
            actions_successful=7,
            evacuation_ordered=True,
            evacuation_rate=0.85,
            lead_time_hours=18,
            lead_time_used_hours=12,
            false_alarm=False,
            confidence_rating=0.8
        )
        
        async def test_feedback():
            return await feedback.record_outcome(outcome)
        
        result = asyncio.run(test_feedback())
        
        if result["recorded"]:
            print_check("Outcome recording", True, "Outcome recorded successfully")
            results.append(True)
        else:
            print_check("Outcome recording", False, "Failed to record")
            results.append(False)
        
        # Check metrics calculation
        metrics = feedback.calculate_system_metrics()
        if metrics is not None:
            print_check("Metrics calculation", True, f"Accuracy: {metrics.accuracy:.1%}")
            results.append(True)
        else:
            print_check("Metrics calculation", False, "No metrics")
            results.append(False)
            
    except Exception as e:
        print_check("Feedback System", False, str(e))
        results.extend([False, False])
    
    # =========================================================================
    # Check 8: Full Pipeline Integration
    # =========================================================================
    print_header("8. Full Pipeline Integration")
    
    try:
        from services.alert_formatter import AlertFormatter
        from services.action_generator import ActionGenerator
        from services.resource_calculator import ResourceCalculator
        from services.alert_distributor import AlertDistributor
        from services.emergency_system_integrator import EmergencySystemIntegrator
        
        # Run full pipeline
        formatter = AlertFormatter()
        generator = ActionGenerator()
        calculator = ResourceCalculator()
        distributor = AlertDistributor()
        integrator = EmergencySystemIntegrator()
        
        # Step 1: Format
        alert = formatter.format_prediction(prediction, location_data)
        
        # Step 2: Generate actions
        action_plan = generator.generate_actions(prediction, location_data)
        
        # Step 3: Calculate resources
        resources = calculator.calculate_resources(prediction, location_data)
        
        # Step 4: Distribute
        async def run_pipeline():
            recipients = [{"type": "authority", "identifier": "test@test.com"}]
            dist = await distributor.distribute_alert(alert.to_simple_dict(), recipients)
            
            # Step 5: Integrate
            integ = await integrator.integrate_alert(
                alert.to_dict(),
                [a.to_dict() for a in action_plan.actions],
                resources.to_dict()
            )
            return dist, integ
        
        dist_result, int_result = asyncio.run(run_pipeline())
        
        if alert.identifier and len(action_plan.actions) > 0 and len(resources.resources) > 0:
            print_check("Pipeline execution", True, "All stages completed")
            results.append(True)
        else:
            print_check("Pipeline execution", False, "Pipeline incomplete")
            results.append(False)
        
        # Check data consistency
        if alert.hazard_type == action_plan.hazard_type == resources.hazard_type:
            print_check("Data consistency", True, "Hazard type consistent")
            results.append(True)
        else:
            print_check("Data consistency", False, "Data mismatch")
            results.append(False)
            
    except Exception as e:
        print_check("Full Pipeline", False, str(e))
        results.extend([False, False])
    
    # =========================================================================
    # Check 9: Documentation
    # =========================================================================
    print_header("9. Documentation")
    
    docs_path = project_root / "docs" / "PHASE3_ALERT_SYSTEM.md"
    if docs_path.exists():
        try:
            with open(docs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if len(content) > 1000:
                print_check("Documentation exists", True, f"{len(content)} characters")
                results.append(True)
            else:
                print_check("Documentation exists", False, "Too short")
                results.append(False)
        except Exception as e:
            print_check("Documentation exists", False, f"Read error: {e}")
            results.append(False)
    else:
        print_check("Documentation exists", False, "File not found")
        results.append(False)
    
    # =========================================================================
    # Check 10: Tests
    # =========================================================================
    print_header("10. Test Suite")
    
    test_path = project_root / "tests" / "test_phase3_alerts.py"
    if test_path.exists():
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
            test_count = content.count("def test_")
            if test_count >= 30:
                print_check("Test suite", True, f"{test_count} tests defined")
                results.append(True)
            else:
                print_check("Test suite", False, f"Only {test_count} tests")
                results.append(False)
        except Exception as e:
            print_check("Test suite", False, f"Read error: {e}")
            results.append(False)
    else:
        print_check("Test suite", False, "File not found")
        results.append(False)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_header("VALIDATION SUMMARY")
    
    passed = sum(results)
    total = len(results)
    pct = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n  Total Checks: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    print(f"  Success Rate: {pct:.1f}%")
    
    if passed == total:
        print("\n  [SUCCESS] Phase 3 implementation complete!")
        return 0
    elif pct >= 80:
        print("\n  [WARNING] Phase 3 mostly complete, some issues remain")
        return 1
    else:
        print("\n  [FAILURE] Phase 3 has significant issues")
        return 2


if __name__ == "__main__":
    sys.exit(main())
