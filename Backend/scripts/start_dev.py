#!/usr/bin/env python3
"""
Development startup script for Crisis Connect API
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup development environment"""
    print("🔧 Setting up development environment...")
    
    # Create .env from dev.env if it doesn't exist
    if not Path(".env").exists():
        if Path("dev.env").exists():
            print("📋 Creating .env from dev.env template...")
            with open("dev.env", "r") as src, open(".env", "w") as dst:
                dst.write(src.read())
            print("✅ Environment file created")
        else:
            print("⚠️  No dev.env template found. Using defaults.")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

def check_services():
    """Check if required services are running"""
    print("🔍 Checking services...")
    
    # Check MongoDB
    try:
        import pymongo
        client = pymongo.MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
        client.server_info()
        print("✅ MongoDB is running")
        client.close()
    except Exception:
        print("⚠️  MongoDB is not running")
        print("   Start with: mongod (or use Docker)")
    
    # Check Redis (optional)
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=5)
        r.ping()
        print("✅ Redis is running")
        r.close()
    except Exception:
        print("⚠️  Redis is not running (optional)")
        print("   Start with: redis-server (or use Docker)")

def start_server():
    """Start the development server"""
    print("🚀 Starting Crisis Connect API server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation: http://localhost:8000/docs")
    print("🏥 Health check: http://localhost:8000/health")
    print("📊 Metrics: http://localhost:8000/metrics")
    print("\n🛑 Press Ctrl+C to stop the server\n")

    try:
        import uvicorn
        # Ensure we're in the correct directory for imports
        script_dir = Path(__file__).parent.parent  # Go up from scripts/ to Backend/
        os.chdir(script_dir)
        # Add the current directory to Python path
        sys.path.insert(0, str(script_dir))
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("🌟 Crisis Connect API - Development Server")
    print("=" * 50)
    
    setup_environment()
    install_dependencies()
    check_services()
    start_server()

if __name__ == "__main__":
    main()
