#!/usr/bin/env python3
"""
Crisis Connect Dashboard Startup Script
Launches the modern Streamlit admin dashboard
"""
import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly"], check=True)
        print("✅ Streamlit installed")
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "plotly"], check=True)
        print("✅ Plotly installed")

def check_api_connection():
    """Check if the API is running"""
    print("🔍 Checking API connection...")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and accessible")
            return True
        else:
            print("⚠️ API responded with status:", response.status_code)
            return False
    except requests.exceptions.RequestException:
        print("⚠️ API is not running or not accessible")
        print("   Start the API with: python main.py")
        return False

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("🚀 Starting Crisis Connect Dashboard...")
    print("📍 Dashboard will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Change to the Backend directory
        backend_dir = Path(__file__).parent.parent
        os.chdir(backend_dir)
        
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("🌟 Crisis Connect - Modern Admin Dashboard")
    print("=" * 50)
    
    check_dependencies()
    
    # Check API connection but don't fail if it's not running
    api_running = check_api_connection()
    if not api_running:
        print("⚠️ Note: Some features may not work without the API running")
        print("   Start the API in another terminal: python main.py")
        print()
    
    start_dashboard()

if __name__ == "__main__":
    main()
