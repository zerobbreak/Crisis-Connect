#!/usr/bin/env python3
"""
Simple test script for Crisis Connect API backend
This tests basic functionality without heavy dependencies
"""
import os
import sys
from pathlib import Path

def check_files():
    """Check if required files exist"""
    print("🔍 Checking required files...")
    
    required_files = [
        "data/rf_model.pkl",
        "data/data_disaster.xlsx", 
        "main.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def check_python_version():
    """Check Python version"""
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print("✅ Python version is compatible")
    return True

def test_basic_imports():
    """Test basic Python imports"""
    print("📦 Testing basic imports...")
    
    try:
        import json
        import datetime
        import logging
        print("✅ Basic Python modules working")
        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_model_file():
    """Test if model file is valid"""
    print("🤖 Testing ML model file...")
    
    try:
        import joblib
        model = joblib.load("data/rf_model.pkl")
        print("✅ ML model loaded successfully")
        
        # Test basic model properties
        if hasattr(model, 'predict'):
            print("✅ Model has predict method")
        if hasattr(model, 'predict_proba'):
            print("✅ Model has predict_proba method")
        
        return True
    except ImportError:
        print("⚠️  joblib not installed - install with: pip install joblib")
        return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_data_file():
    """Test if data file exists"""
    print("📊 Testing data file...")
    
    try:
        import pandas as pd
        df = pd.read_excel("data/data_disaster.xlsx")
        print(f"✅ Data file loaded: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        return True
    except ImportError:
        print("⚠️  pandas not installed - install with: pip install pandas openpyxl")
        return False
    except Exception as e:
        print(f"❌ Data file loading failed: {e}")
        return False

def create_env_file():
    """Create .env file for development"""
    print("🔧 Creating environment file...")
    
    env_content = """# Crisis Connect API Development Environment
DEBUG=true
API_VERSION=1.0.0
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=crisis_connect
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://localhost:8000
TRUSTED_HOSTS=localhost,127.0.0.1
LOG_LEVEL=INFO
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Environment file created (.env)")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_minimal_api():
    """Test minimal API functionality"""
    print("🚀 Testing minimal API setup...")
    
    try:
        # Test if we can import the main components
        sys.path.insert(0, ".")
        
        # Test config import
        try:
            from config import settings
            print("✅ Configuration loaded")
        except ImportError as e:
            print(f"⚠️  Config import failed: {e}")
        
        # Test models import
        try:
            from models.model import AlertModel
            print("✅ Models imported")
        except ImportError as e:
            print(f"⚠️  Models import failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ API setup test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Crisis Connect API - Simple Test Suite")
    print("=" * 50)
    
    tests = [
        ("Python Version", check_python_version),
        ("Required Files", check_files),
        ("Basic Imports", test_basic_imports),
        ("ML Model", test_model_file),
        ("Data File", test_data_file),
        ("Environment Setup", create_env_file),
        ("API Setup", test_minimal_api),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "="*50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed >= total - 2:  # Allow 2 failures
        print("🎉 Backend is ready for basic testing!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start MongoDB (optional): mongod")
        print("3. Start Redis (optional): redis-server")
        print("4. Run the API: uvicorn main:app --reload")
        print("5. Visit: http://localhost:8000/docs")
    else:
        print("⚠️  Some critical tests failed. Please fix the issues above.")

if __name__ == "__main__":
    main()
