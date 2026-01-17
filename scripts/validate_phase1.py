"""
Phase 1 Validation Script

One-command verification that Phase 1 Data Foundation is complete.
Checks all success criteria are met.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_master_dataset():
    """Check master dataset exists"""
    return (project_root / "data" / "processed" / "disasters_master.csv").exists()


def check_record_count():
    """Check minimum record count (1000+)"""
    if not check_master_dataset():
        return False, 0
    
    try:
        import pandas as pd
        df = pd.read_csv(project_root / "data" / "processed" / "disasters_master.csv")
        return len(df) >= 1000, len(df)
    except Exception as e:
        return False, 0


def check_quality_score():
    """Check data quality score >= 70"""
    report_path = project_root / "data" / "quality_reports.jsonl"
    if not report_path.exists():
        return False, 0
    
    try:
        with open(report_path, "r") as f:
            lines = f.readlines()
        
        if not lines:
            return False, 0
        
        latest = json.loads(lines[-1])
        score = latest.get("overall_score", 0)
        return score >= 70, score
    except Exception as e:
        return False, 0


def check_features():
    """Check feature count >= 30"""
    if not check_master_dataset():
        return False, 0
    
    try:
        import pandas as pd
        df = pd.read_csv(project_root / "data" / "processed" / "disasters_master.csv")
        return len(df.columns) >= 30, len(df.columns)
    except Exception as e:
        return False, 0


def check_quality_reports():
    """Check quality reports exist"""
    return (project_root / "data" / "quality_reports.jsonl").exists()


def check_scheduler():
    """Check scheduler script exists"""
    return (project_root / "scripts" / "schedule_data_collection.py").exists()


def check_tests():
    """Check test file exists"""
    return (project_root / "tests" / "test_phase1_data_pipeline.py").exists()


def check_documentation():
    """Check documentation exists"""
    return (project_root / "docs" / "PHASE1_DATA_DOCUMENTATION.md").exists()


def check_run_instructions():
    """Check run instructions exist"""
    return (project_root / "PHASE1_RUN_INSTRUCTIONS.md").exists()


def check_services():
    """Check all required service files exist"""
    services = [
        "data_sources/emdat_fetcher.py",
        "data_sources/noaa_fetcher.py",
        "data_cleaning.py",
        "data_validation.py",
        "feature_engineering.py",
        "data_quality_monitor.py",
        "data_storage.py",
    ]
    
    services_dir = project_root / "services"
    missing = []
    
    for service in services:
        if not (services_dir / service).exists():
            missing.append(service)
    
    return len(missing) == 0, missing


def run_tests():
    """Run pytest for Phase 1 tests"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", 
             str(project_root / "tests" / "test_phase1_data_pipeline.py"),
             "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=120
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def validate_phase1():
    """Comprehensive Phase 1 validation"""
    
    print("=" * 60)
    print("PHASE 1 DATA FOUNDATION - VALIDATION")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Project: {project_root}")
    print("-" * 60)
    
    checks = {}
    details = {}
    
    # Infrastructure checks
    print("\n[INFRASTRUCTURE]")
    
    services_ok, missing = check_services()
    checks["All service files exist"] = services_ok
    if not services_ok:
        details["services"] = f"Missing: {missing}"
    print(f"  [{'PASS' if services_ok else 'FAIL'}] All service files exist")
    
    checks["Scheduler configured"] = check_scheduler()
    print(f"  [{'PASS' if checks['Scheduler configured'] else 'FAIL'}] Scheduler configured")
    
    checks["Tests exist"] = check_tests()
    print(f"  [{'PASS' if checks['Tests exist'] else 'FAIL'}] Tests exist")
    
    # Documentation checks
    print("\n[DOCUMENTATION]")
    
    checks["Run instructions exist"] = check_run_instructions()
    print(f"  [{'PASS' if checks['Run instructions exist'] else 'FAIL'}] Run instructions exist")
    
    checks["Data documentation exists"] = check_documentation()
    print(f"  [{'PASS' if checks['Data documentation exists'] else 'FAIL'}] Data documentation exists")
    
    # Data checks
    print("\n[DATA]")
    
    checks["Master dataset exists"] = check_master_dataset()
    print(f"  [{'PASS' if checks['Master dataset exists'] else 'FAIL'}] Master dataset exists")
    
    record_ok, record_count = check_record_count()
    checks["Minimum 1000 records"] = record_ok
    details["records"] = record_count
    print(f"  [{'PASS' if record_ok else 'WARN'}] Minimum 1000 records (actual: {record_count})")
    
    feature_ok, feature_count = check_features()
    checks["Minimum 30 features"] = feature_ok
    details["features"] = feature_count
    print(f"  [{'PASS' if feature_ok else 'WARN'}] Minimum 30 features (actual: {feature_count})")
    
    # Quality checks
    print("\n[QUALITY]")
    
    checks["Quality reports exist"] = check_quality_reports()
    print(f"  [{'PASS' if checks['Quality reports exist'] else 'FAIL'}] Quality reports exist")
    
    quality_ok, quality_score = check_quality_score()
    checks["Quality score >= 70"] = quality_ok
    details["quality_score"] = quality_score
    print(f"  [{'PASS' if quality_ok else 'WARN'}] Quality score >= 70 (actual: {quality_score:.1f})")
    
    # Test execution
    print("\n[TESTS]")
    if check_tests():
        print("  Running tests...")
        tests_ok, test_output = run_tests()
        checks["All tests pass"] = tests_ok
        if tests_ok:
            print("  [PASS] All tests pass")
        else:
            print("  [FAIL] Some tests failed")
            # Show brief summary
            lines = test_output.split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"      {line}")
    else:
        checks["All tests pass"] = False
        print("  [FAIL] Tests not found")
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    if passed == total:
        print("PHASE 1 COMPLETE!")
        print(f"   All {total} checks passed")
    elif passed >= total - 2:
        print("PHASE 1 NEARLY COMPLETE")
        print(f"   {passed}/{total} checks passed")
        print("\n   Missing/failing:")
        for check, passed in checks.items():
            if not passed:
                print(f"   - {check}")
    else:
        print("PHASE 1 INCOMPLETE")
        print(f"   {passed}/{total} checks passed")
    
    print("=" * 60)
    
    return all(checks.values()), checks, details


if __name__ == "__main__":
    success, checks, details = validate_phase1()
    sys.exit(0 if success else 1)
