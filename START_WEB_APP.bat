@echo off
REM LSTM Speech Recognition Web App Startup Script (Windows)
REM This script starts the Flask web application

echo.
echo ================================================================
echo   LSTM Kinyarwanda Speech Recognition Web App
echo ================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Navigate to web_app directory
cd /d "%~dp0web_app"

REM Check if virtual environment exists
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo.
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if model exists
if not exist "..\models\trained\kinyarwanda_masked_final.h5" (
    if exist "..\models\checkpoints" (
        echo.
        echo WARNING: Training model not found, but checkpoint exists
        echo The app will try to use the latest checkpoint
    ) else (
        echo.
        echo WARNING: No trained model found
        echo Please run training first with: python ..\scripts\train_with_masking.py
    )
)

REM Start the Flask app
echo.
echo ================================================================
echo   Starting Flask Web Server...
echo   Access the app at: http://localhost:5000
echo ================================================================
echo.

python app.py

pause

