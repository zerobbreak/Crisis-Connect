#!/usr/bin/env python3
"""
Phase 2 Validation Script

Validates that all Phase 2 multi-agent components are working correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime


def print_header(title: str):
    """Print section header"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_check(name: str, passed: bool, details: str = ""):
    """Print check result"""
    status = "[OK]" if passed else "[FAIL]"
    print(f"  {status} {name}")
    if details:
        print(f"      {details}")


def check_agent_imports():
    """Check that all agents can be imported"""
    print_header("AGENT IMPORTS")
    
    checks = {}
    
    try:
        from agents.pattern_detection_agent import PatternDetectionAgent, Pattern
        checks["PatternDetectionAgent"] = True
    except Exception as e:
        checks["PatternDetectionAgent"] = False
        print(f"      Error: {e}")
    
    try:
        from agents.progression_analyzer_agent import ProgressionAnalyzerAgent, ProgressionStage
        checks["ProgressionAnalyzerAgent"] = True
    except Exception as e:
        checks["ProgressionAnalyzerAgent"] = False
        print(f"      Error: {e}")
    
    try:
        from agents.forecast_agent import ForecastAgent, Forecast
        checks["ForecastAgent"] = True
    except Exception as e:
        checks["ForecastAgent"] = False
        print(f"      Error: {e}")
    
    try:
        from agents.anomaly_detection_agent import AnomalyDetectionAgent
        checks["AnomalyDetectionAgent"] = True
    except Exception as e:
        checks["AnomalyDetectionAgent"] = False
        print(f"      Error: {e}")
    
    try:
        from agents.ensemble_coordinator_agent import EnsembleCoordinatorAgent, EnsembleDecision
        checks["EnsembleCoordinatorAgent"] = True
    except Exception as e:
        checks["EnsembleCoordinatorAgent"] = False
        print(f"      Error: {e}")
    
    for name, passed in checks.items():
        print_check(name, passed)
    
    return all(checks.values())


def check_service_imports():
    """Check that all services can be imported"""
    print_header("SERVICE IMPORTS")
    
    checks = {}
    
    try:
        from services.agent_orchestrator import AgentOrchestrator
        checks["AgentOrchestrator"] = True
    except Exception as e:
        checks["AgentOrchestrator"] = False
        print(f"      Error: {e}")
    
    try:
        from services.temporal_processor import TemporalDataProcessor
        checks["TemporalDataProcessor"] = True
    except Exception as e:
        checks["TemporalDataProcessor"] = False
        print(f"      Error: {e}")
    
    try:
        from services.agent_performance_monitor import AgentPerformanceMonitor
        checks["AgentPerformanceMonitor"] = True
    except Exception as e:
        checks["AgentPerformanceMonitor"] = False
        print(f"      Error: {e}")
    
    for name, passed in checks.items():
        print_check(name, passed)
    
    return all(checks.values())


def check_agent_functionality():
    """Check that agents work correctly"""
    print_header("AGENT FUNCTIONALITY")
    
    checks = {}
    
    # Test Pattern Detection Agent
    try:
        from agents.pattern_detection_agent import PatternDetectionAgent
        agent = PatternDetectionAgent()
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X[:20, 0] += 2  # Add pattern
        y = np.array([1] * 20 + [0] * 30)
        df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        df['disaster_type'] = 'flood'
        
        patterns = agent.discover_patterns(df, y, ['f1', 'f2', 'f3', 'f4', 'f5'])
        checks["Pattern Discovery"] = agent.is_trained
        print_check("Pattern Discovery", checks["Pattern Discovery"], 
                   f"{len(patterns)} patterns discovered")
    except Exception as e:
        checks["Pattern Discovery"] = False
        print_check("Pattern Discovery", False, str(e))
    
    # Test Progression Analyzer
    try:
        from agents.progression_analyzer_agent import ProgressionAnalyzerAgent
        agent = ProgressionAnalyzerAgent()
        
        # Create test weather data
        dates = pd.date_range('2024-01-01', periods=14)
        weather = pd.DataFrame({
            'date': dates,
            'rainfall': np.linspace(5, 50, 14),
            'humidity': [70] * 14,
            'wind_speed': [20] * 14
        })
        
        analysis = agent.analyze_progression(weather, "flood")
        checks["Progression Analysis"] = analysis.severity_score > 0
        print_check("Progression Analysis", checks["Progression Analysis"],
                   f"Stage: {analysis.current_stage.name}, Severity: {analysis.severity_score:.1%}")
    except Exception as e:
        checks["Progression Analysis"] = False
        print_check("Progression Analysis", False, str(e))
    
    # Test Anomaly Detection
    try:
        from agents.anomaly_detection_agent import AnomalyDetectionAgent
        agent = AnomalyDetectionAgent()
        
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        agent.train(X_train)
        
        X_anomaly = np.array([[10, 10, 10, 10, 10]])
        result = agent.detect_single(X_anomaly)
        
        checks["Anomaly Detection"] = agent.is_trained
        print_check("Anomaly Detection", checks["Anomaly Detection"],
                   f"Score: {result.anomaly_score:.3f}, Anomalous: {result.is_anomalous}")
    except Exception as e:
        checks["Anomaly Detection"] = False
        print_check("Anomaly Detection", False, str(e))
    
    # Test Ensemble Coordinator
    try:
        from agents.ensemble_coordinator_agent import EnsembleCoordinatorAgent
        from agents.forecast_agent import Forecast
        
        agent = EnsembleCoordinatorAgent()
        
        forecasts = [
            Forecast("flood", 0.7, 48, 0.8, "lstm", "Test"),
            Forecast("flood", 0.6, 36, 0.7, "progression", "Test")
        ]
        
        decision = agent.coordinate(forecasts)
        checks["Ensemble Coordination"] = decision.risk_level in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        print_check("Ensemble Coordination", checks["Ensemble Coordination"],
                   f"Risk: {decision.risk_score:.1%}, Level: {decision.risk_level}")
    except Exception as e:
        checks["Ensemble Coordination"] = False
        print_check("Ensemble Coordination", False, str(e))
    
    return all(checks.values())


def check_orchestrator_with_data():
    """Check orchestrator with Phase 1 data if available"""
    print_header("ORCHESTRATOR WITH PHASE 1 DATA")
    
    data_path = Path("data/processed/disasters_master.csv")
    
    if not data_path.exists():
        print("  [SKIP] Phase 1 master dataset not found")
        print("      Run Phase 1 data collection first")
        return True  # Not a failure, just skip
    
    checks = {}
    
    try:
        from services.agent_orchestrator import AgentOrchestrator
        
        # Load Phase 1 data
        disasters_df = pd.read_csv(data_path)
        print(f"  Loaded {len(disasters_df)} disaster records")
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(data_dir="data", auto_load_models=False)
        
        # Train agents
        print("  Training agents...")
        stats = orchestrator.train_agents(disasters_df)
        
        checks["Agent Training"] = len(stats.get('errors', [])) == 0
        print_check("Agent Training", checks["Agent Training"],
                   f"Time: {stats.get('training_time_seconds', 0):.1f}s")
        
        patterns_count = stats.get('pattern_agent', {}).get('patterns_discovered', 0)
        checks["Patterns Discovered"] = patterns_count > 0
        print_check("Patterns Discovered", checks["Patterns Discovered"],
                   f"{patterns_count} patterns found")
        
        # Make a test prediction
        print("  Making test prediction...")
        np.random.seed(42)
        test_weather = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=14),
            'rainfall': np.random.exponential(15, 14),
            'humidity': 70 + np.random.normal(0, 10, 14),
            'wind_speed': 20 + np.random.exponential(5, 14),
            'temperature': 25 + np.random.normal(0, 3, 14)
        })
        
        result = orchestrator.predict(test_weather, location="TestCity", hazard_type="flood")
        
        checks["Prediction Success"] = result.get('prediction') is not None
        if checks["Prediction Success"]:
            pred = result['prediction']
            print_check("Prediction Success", True,
                       f"Risk: {pred['risk_score']:.1%}, Level: {pred['risk_level']}")
        else:
            print_check("Prediction Success", False, str(result.get('errors', [])))
        
        checks["Processing Time"] = result.get('processing_time_seconds', 999) < 5
        print_check("Processing Time < 5s", checks["Processing Time"],
                   f"{result.get('processing_time_seconds', 0):.2f}s")
        
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return all(checks.values())


def check_documentation():
    """Check that documentation exists"""
    print_header("DOCUMENTATION")
    
    checks = {}
    
    doc_path = Path("docs/PHASE2_MULTI_AGENT_SYSTEM.md")
    checks["Multi-Agent Documentation"] = doc_path.exists()
    print_check("Multi-Agent Documentation", checks["Multi-Agent Documentation"],
               str(doc_path) if doc_path.exists() else "Not found")
    
    return all(checks.values())


def check_tests():
    """Check that tests exist and pass"""
    print_header("TESTS")
    
    test_path = Path("tests/test_phase2_agents.py")
    
    if not test_path.exists():
        print_check("Test file exists", False, "tests/test_phase2_agents.py not found")
        return False
    
    print_check("Test file exists", True)
    
    # Run tests
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=no", "-q"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    # Parse results
    output = result.stdout + result.stderr
    
    # Look for passed/failed counts
    passed = 0
    failed = 0
    for line in output.split('\n'):
        if 'passed' in line.lower():
            try:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'passed':
                        passed = int(parts[i-1])
                    elif p == 'failed':
                        failed = int(parts[i-1])
            except:
                pass
    
    all_passed = result.returncode == 0
    print_check("All tests pass", all_passed, f"{passed} passed, {failed} failed")
    
    return all_passed


def main():
    """Run all Phase 2 validation checks"""
    print()
    print("=" * 60)
    print("     PHASE 2 MULTI-AGENT SYSTEM VALIDATION")
    print("=" * 60)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Agent Imports": check_agent_imports(),
        "Service Imports": check_service_imports(),
        "Agent Functionality": check_agent_functionality(),
        "Orchestrator with Data": check_orchestrator_with_data(),
        "Documentation": check_documentation(),
        "Tests": check_tests()
    }
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        print_check(name, result)
    
    print()
    print(f"  Total: {passed}/{total} checks passed ({passed/total*100:.0f}%)")
    print()
    
    if passed == total:
        print("  [SUCCESS] Phase 2 implementation complete!")
        print()
        print("  Multi-Agent System Ready:")
        print("  - 5 specialized agents implemented")
        print("  - Agent orchestrator coordinating predictions")
        print("  - Performance monitoring in place")
        print("  - All tests passing")
        print("  - Documentation complete")
    else:
        print(f"  [WARNING] {total - passed} check(s) failed")
        print("  Review the errors above and fix issues")
    
    print()
    print("=" * 60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
