#!/bin/bash
# LSTM Speech Recognition Web App Startup Script (Linux/Mac)
# This script starts the Flask web application

echo ""
echo "================================================================"
echo "   LSTM Kinyarwanda Speech Recognition Web App"
echo "================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from python.org"
    exit 1
fi

# Navigate to web_app directory
cd "$(dirname "$0")/web_app"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo ""
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "../models/trained/kinyarwanda_masked_final.h5" ]; then
    if [ -d "../models/checkpoints" ]; then
        echo ""
        echo "WARNING: Training model not found, but checkpoint exists"
        echo "The app will try to use the latest checkpoint"
    else
        echo ""
        echo "WARNING: No trained model found"
        echo "Please run training first with: python ../scripts/train_with_masking.py"
    fi
fi

# Start the Flask app
echo ""
echo "================================================================"
echo "   Starting Flask Web Server..."
echo "   Access the app at: http://localhost:5000"
echo "================================================================"
echo ""

python app.py

