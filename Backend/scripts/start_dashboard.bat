@echo off
echo Crisis Connect Dashboard Startup
echo ===============================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Installing dependencies...
pip install streamlit plotly requests

REM Start the dashboard
echo Starting Crisis Connect Dashboard...
echo Dashboard will be available at: http://localhost:8501
echo Press Ctrl+C to stop
echo.

python scripts/start_dashboard.py

pause
