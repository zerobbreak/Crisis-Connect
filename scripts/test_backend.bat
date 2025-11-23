@echo off
echo Crisis Connect API Backend Test
echo ===============================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    if exist dev.env (
        copy dev.env .env
        echo Created .env file from template
    )
)

REM Run the test script
echo Running backend tests...
python test_backend.py

pause
