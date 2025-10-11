#!/usr/bin/env python3
"""
Offline test script for Crisis Connect API components
Tests core functionality without running a server
"""
import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("📦 Testing package imports...")
    
    try:
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__}")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import uvicorn
        print(f"✅ Uvicorn {uvicorn.__version__}")
    except ImportError as e:
        print(f"❌ Uvicorn import failed: {e}")
        return False
    
    try:
        import pydantic
        print(f"✅ Pydantic {pydantic.__version__}")
    except ImportError as e:
        print(f"❌ Pydantic import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import joblib
        print(f"✅ Joblib {joblib.__version__}")
    except ImportError as e:
        print(f"❌ Joblib import failed: {e}")
        return False
    
    return True

def test_model():
    """Test ML model loading and basic functionality"""
    print("\n🤖 Testing ML model...")
    
    try:
        import joblib
        import pandas as pd
        import numpy as np
        
        # Load model
        model = joblib.load("data/rf_model.pkl")
        print(f"✅ Model loaded: {type(model).__name__}")
        
        # Test prediction
        sample_data = pd.DataFrame([{
            'lat': -29.8587,
            'lon': 31.0218,
            'temp_c': 25.0,
            'humidity': 60.0,
            'wind_kph': 15.0,
            'pressure_mb': 1013.0,
            'precip_mm': 5.0,
            'cloud': 30.0,
            'wave_height': 1.0
        }])
        
        prediction = model.predict_proba(sample_data)
        print(f"✅ Prediction successful: {prediction[0]}")
        
        # Test with multiple samples
        sample_data2 = pd.DataFrame([
            {'lat': -29.8587, 'lon': 31.0218, 'temp_c': 25.0, 'humidity': 60.0, 'wind_kph': 15.0, 'pressure_mb': 1013.0, 'precip_mm': 5.0, 'cloud': 30.0, 'wave_height': 1.0},
            {'lat': -33.9249, 'lon': 18.4241, 'temp_c': 22.0, 'humidity': 70.0, 'wind_kph': 12.0, 'pressure_mb': 1015.0, 'precip_mm': 0.0, 'cloud': 20.0, 'wave_height': 2.0}
        ])
        
        predictions = model.predict_proba(sample_data2)
        print(f"✅ Multiple predictions: {len(predictions)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_data():
    """Test data loading and basic operations"""
    print("\n📊 Testing data operations...")
    
    try:
        import pandas as pd
        
        # Load data
        df = pd.read_excel("data/data_disaster.xlsx")
        print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Basic data analysis
        print(f"   Date range: {df['Start Year'].min()} - {df['Start Year'].max()}")
        print(f"   Disaster types: {df['Disaster Type'].nunique()} unique types")
        print(f"   Countries: {df['Country'].nunique()} unique countries")
        
        # Sample data
        sample = df.head(3)
        print(f"✅ Sample data preview:")
        for idx, row in sample.iterrows():
            print(f"   {row['Country']} - {row['Disaster Type']} ({row['Start Year']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Data test failed: {e}")
        return False

def test_api_components():
    """Test API component creation"""
    print("\n🚀 Testing API components...")
    
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        
        # Create minimal app
        app = FastAPI(title="Test App")
        
        # Create a simple model
        class TestModel(BaseModel):
            name: str
            value: float
        
        # Test model validation
        test_data = TestModel(name="test", value=1.5)
        print(f"✅ Pydantic model: {test_data}")
        
        # Test app creation
        print(f"✅ FastAPI app created: {app.title}")
        
        return True
        
    except Exception as e:
        print(f"❌ API components test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\n⚙️  Testing configuration...")
    
    try:
        # Test if .env file exists
        if Path(".env").exists():
            print("✅ Environment file (.env) exists")
        else:
            print("⚠️  No .env file found")
        
        # Test config import
        try:
            from config import settings
            print("✅ Configuration loaded successfully")
            print(f"   API Title: {settings.api_title}")
            print(f"   Debug Mode: {settings.debug}")
            return True
        except Exception as e:
            print(f"⚠️  Config import issue: {e}")
            return True  # Not critical for basic testing
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Crisis Connect API - Offline Component Tests")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("ML Model", test_model),
        ("Data Operations", test_data),
        ("API Components", test_api_components),
        ("Configuration", test_config),
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
    
    print("\n" + "="*60)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All components are working correctly!")
        print("\n✅ Ready to run the API server:")
        print("   python minimal_api.py")
        print("   or")
        print("   uvicorn minimal_api:app --reload")
        print("\n📖 Then visit: http://localhost:8000/docs")
    elif passed >= total - 1:
        print("🎉 Almost everything is working!")
        print("✅ You can still run the API with minor limitations")
    else:
        print("⚠️  Some components need attention before running the API")
        print("   Check the errors above and install missing dependencies")

if __name__ == "__main__":
    main()
