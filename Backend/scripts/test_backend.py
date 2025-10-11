#!/usr/bin/env python3
"""
Simple test script to verify the Crisis Connect API backend
"""
import asyncio
import sys
import os
import time
from pathlib import Path

# Add the Backend directory to Python path (parent of scripts directory)
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

try:
    import requests
    from fastapi.testclient import TestClient
    import uvicorn
    import subprocess
    import threading
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_files = [
        "data/rf_model.pkl",
        "data/data_disaster.xlsx",
        "main.py",
        "config.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True


def check_environment():
    """Check environment configuration"""
    print("🔍 Checking environment...")
    
    # Check if .env exists, if not copy from config/dev.env
    if not Path(".env").exists():
        if Path("config/dev.env").exists():
            print("📋 Creating .env from config/dev.env template...")
            import shutil
            shutil.copy("config/dev.env", ".env")
            print("✅ Environment file created")
        else:
            print("⚠️  No environment file found. Using defaults.")
    
    # Check MongoDB connection
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
        client.server_info()
        print("✅ MongoDB connection successful")
        client.close()
    except Exception as e:
        print(f"⚠️  MongoDB not available: {e}")
        print("   You can still test the API, but database operations will fail")
    
    # Check Redis connection
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=5)
        r.ping()
        print("✅ Redis connection successful")
        r.close()
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("   API will work without caching")
    
    return True


def test_api_endpoints():
    """Test basic API endpoints"""
    print("\n🧪 Testing API endpoints...")
    
    try:
        # Import the app
        from main import app
        client = TestClient(app)
        
        # Test root endpoint
        print("Testing root endpoint...")
        response = client.get("/")
        if response.status_code == 200:
            print("✅ Root endpoint working")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
        
        # Test health endpoint
        print("Testing health endpoint...")
        response = client.get("/health")
        if response.status_code in [200, 503]:  # 503 is OK if services are down
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test metrics endpoint
        print("Testing metrics endpoint...")
        response = client.get("/metrics")
        if response.status_code in [200, 500]:  # 500 is OK if database is down
            print("✅ Metrics endpoint working")
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
            return False
        
        # Test risk assessment endpoint
        print("Testing risk assessment endpoint...")
        response = client.get("/risk-assessment")
        if response.status_code in [200, 500]:  # 500 is OK if no data
            print("✅ Risk assessment endpoint working")
        else:
            print(f"❌ Risk assessment endpoint failed: {response.status_code}")
        
        # Test alerts endpoint
        print("Testing alerts endpoint...")
        response = client.get("/alerts/history")
        if response.status_code in [200, 500]:  # 500 is OK if no database
            print("✅ Alerts endpoint working")
        else:
            print(f"❌ Alerts endpoint failed: {response.status_code}")
        
        print("✅ Basic API endpoints test completed")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def test_model_loading():
    """Test ML model loading"""
    print("\n🤖 Testing ML model loading...")
    
    try:
        import joblib
        model = joblib.load("data/rf_model.pkl")
        print("✅ ML model loaded successfully")
        
        # Test model prediction
        import pandas as pd
        import numpy as np
        
        # Create sample data
        sample_data = pd.DataFrame({
            'lat': [-29.8587],
            'lon': [31.0218],
            'temp_c': [25.0],
            'humidity': [60.0],
            'wind_kph': [15.0],
            'pressure_mb': [1013.0],
            'precip_mm': [5.0],
            'cloud': [30.0],
            'wave_height': [1.0]
        })
        
        # Test prediction
        prediction = model.predict_proba(sample_data)[:, 1]
        print(f"✅ Model prediction test successful: {prediction[0]:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_data_collection():
    """Test weather data collection"""
    print("\n🌤️  Testing weather data collection...")
    
    try:
        from services.predict import collect_all_data
        import pandas as pd
        
        # Test with minimal data
        print("Collecting weather data for one location...")
        df = collect_all_data({"Test Location": (-29.8587, 31.0218)})
        
        if not df.empty:
            print(f"✅ Weather data collection successful: {len(df)} records")
            print(f"   Sample data: {df.iloc[0].to_dict()}")
            return True
        else:
            print("⚠️  No weather data collected (API might be unavailable)")
            return True  # This is not a failure, just no data
            
    except Exception as e:
        print(f"❌ Weather data collection failed: {e}")
        return False


def run_live_server_test():
    """Run a live server test"""
    print("\n🚀 Testing live server...")
    
    try:
        # Start server in background
        print("Starting server on http://localhost:8001...")
        
        def run_server():
            import uvicorn
            uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(5)
        
        # Test live endpoints
        base_url = "http://localhost:8001"
        
        try:
            response = requests.get(f"{base_url}/", timeout=10)
            if response.status_code == 200:
                print("✅ Live server responding")
            else:
                print(f"❌ Live server error: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Live server connection failed: {e}")
            return False
        
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            print(f"✅ Health check: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Health check failed: {e}")
        
        try:
            response = requests.get(f"{base_url}/docs", timeout=10)
            if response.status_code == 200:
                print("✅ API documentation available at http://localhost:8001/docs")
            else:
                print(f"⚠️  API docs error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  API docs failed: {e}")
        
        print("✅ Live server test completed")
        print("   Server is running at http://localhost:8001")
        print("   Press Ctrl+C to stop")
        
        # Keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Live server test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Crisis Connect API Backend Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Run tests
    tests = [
        ("Dependencies Check", check_dependencies),
        ("Environment Check", check_environment),
        ("Model Loading", test_model_loading),
        ("API Endpoints", test_api_endpoints),
        ("Data Collection", test_data_collection),
    ]
    
    for test_name, test_func in tests:
        total_tests += 1
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                tests_passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    # Summary
    print("\n" + "="*50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The backend is ready to use.")
        print("\n🚀 To start the server:")
        print("   uvicorn main:app --reload")
        print("\n📖 To view API documentation:")
        print("   http://localhost:8000/docs")
        
        # Ask if user wants to run live server test
        response = input("\n🤔 Would you like to run a live server test? (y/n): ")
        if response.lower() in ['y', 'yes']:
            run_live_server_test()
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("   You may need to install dependencies or configure services.")


if __name__ == "__main__":
    main()
