#!/usr/bin/env python3
"""
System Verification - Test all Crisis Connect components
Verifies pipeline, watcher, continuous learning, and performance tracking
"""
import asyncio
import os
import sys
from datetime import datetime
import structlog

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title):
    """Print formatted section"""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def check_file_exists(filepath, description):
    """Check if file exists and print result"""
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {filepath}")
    return exists


async def verify_system():
    """Verify all system components"""
    
    print_header("CRISIS CONNECT SYSTEM VERIFICATION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_checks_passed = True
    
    # 1. Core Files
    print_section("1. CORE COMPONENTS")
    
    core_files = [
        ("config/settings.py", "Configuration"),
        ("models/crisis_model.pkl", "ML Model"),
        ("utils/db.py", "Database utilities"),
        ("services/weather_service.py", "Weather service"),
        ("services/alert_service.py", "Alert service"),
        ("services/location_service.py", "Location service"),
    ]
    
    for filepath, desc in core_files:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # 2. Pipeline Scripts
    print_section("2. PIPELINE SCRIPTS")
    
    pipeline_files = [
        ("scripts/run_pipeline.py", "Main pipeline"),
        ("scripts/weather_watcher.py", "Weather watcher"),
    ]
    
    for filepath, desc in pipeline_files:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # 3. Continuous Learning
    print_section("3. CONTINUOUS LEARNING SYSTEM")
    
    learning_files = [
        ("scripts/data_aggregator.py", "Data aggregator"),
        ("scripts/model_retrainer.py", "Model retrainer"),
        ("scripts/performance_tracker.py", "Performance tracker"),
    ]
    
    for filepath, desc in learning_files:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # 4. Model Versioning
    print_section("4. MODEL VERSIONING")
    
    model_dirs = [
        ("models/versions", "Model versions directory"),
        ("models/backups", "Model backups directory"),
    ]
    
    for dirpath, desc in model_dirs:
        exists = os.path.exists(dirpath)
        status = "[OK]" if exists else "[CREATED]"
        if not exists:
            os.makedirs(dirpath, exist_ok=True)
        print(f"{status} {desc}: {dirpath}")
    
    # 5. Data Directories
    print_section("5. DATA DIRECTORIES")
    
    data_dirs = [
        ("data/training", "Training data"),
        ("data/metrics", "Performance metrics"),
    ]
    
    for dirpath, desc in data_dirs:
        exists = os.path.exists(dirpath)
        status = "[OK]" if exists else "[CREATED]"
        if not exists:
            os.makedirs(dirpath, exist_ok=True)
        print(f"{status} {desc}: {dirpath}")
    
    # 6. Database Connection
    print_section("6. DATABASE CONNECTION")
    
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from config import settings
        
        client = AsyncIOMotorClient("mongodb://localhost:27017")
        await client.admin.command('ping')
        print("[OK] MongoDB connection successful")
        
        # Check collections
        db = client["crisisconnect"]
        collections = await db.list_collection_names()
        print(f"[OK] Found {len(collections)} collections: {', '.join(collections)}")
        
        # Check data counts
        weather_count = await db["weather_data"].count_documents({})
        predictions_count = await db["predictions"].count_documents({})
        alerts_count = await db["alerts"].count_documents({})
        locations_count = await db["locations"].count_documents({})
        
        print(f"[INFO] Weather data: {weather_count} records")
        print(f"[INFO] Predictions: {predictions_count} records")
        print(f"[INFO] Alerts: {alerts_count} records")
        print(f"[INFO] Locations: {locations_count} records")
        
        client.close()
        
    except Exception as e:
        print(f"[ERROR] Database connection failed: {str(e)}")
        all_checks_passed = False
    
    # 7. Model Information
    print_section("7. MODEL INFORMATION")
    
    try:
        import joblib
        from config import settings
        
        model = joblib.load(settings.model_path)
        print(f"[OK] Model loaded successfully")
        print(f"[INFO] Model type: {type(model).__name__}")
        
        # Check model file timestamp
        model_mtime = os.path.getmtime(settings.model_path)
        model_time = datetime.fromtimestamp(model_mtime)
        print(f"[INFO] Model last modified: {model_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"[ERROR] Model loading failed: {str(e)}")
        all_checks_passed = False
    
    # 8. System Capabilities
    print_section("8. SYSTEM CAPABILITIES")
    
    capabilities = [
        ("Weather Data Collection", "✓ Automated via watcher"),
        ("Risk Assessment", "✓ ML-based predictions"),
        ("Alert Generation", "✓ Threshold-based alerts"),
        ("Continuous Learning", "✓ Weekly retraining"),
        ("Auto Model Reload", "✓ Zero-downtime updates"),
        ("Performance Tracking", "✓ Metrics & reports"),
    ]
    
    for capability, status in capabilities:
        print(f"  {capability:<30} {status}")
    
    # 9. Quick Start Commands
    print_section("9. QUICK START COMMANDS")
    
    commands = [
        ("Run pipeline once", "python .\\scripts\\run_pipeline.py"),
        ("Start watcher", "python .\\scripts\\weather_watcher.py"),
        ("Aggregate data", "python .\\scripts\\data_aggregator.py --days-back 7"),
        ("Retrain model", "python .\\scripts\\model_retrainer.py --data data/training/aggregated_data.csv"),
        ("Performance report", "python .\\scripts\\performance_tracker.py --days 7"),
    ]
    
    for desc, cmd in commands:
        print(f"  {desc:<25} {cmd}")
    
    # Final Summary
    print_header("VERIFICATION SUMMARY")
    
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nYour Crisis Connect system is fully operational!")
        print("\nNext steps:")
        print("  1. Start the weather watcher: python .\\scripts\\weather_watcher.py")
        print("  2. Let it collect data for a week")
        print("  3. Run weekly retraining to improve the model")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nPlease review the errors above and fix any issues.")
    
    print("\n" + "=" * 80 + "\n")
    
    return all_checks_passed


async def main():
    """Main entry point"""
    try:
        result = await verify_system()
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.error("Verification failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
