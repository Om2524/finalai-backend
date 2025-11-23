#!/bin/bash

# Ask Doubt Backend Startup Script

echo "=========================================="
echo "Starting Ask Doubt Backend"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if Manim is installed
if ! command -v manim &> /dev/null; then
    echo "WARNING: Manim command not found. Make sure Manim is installed."
fi

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg not found. Please install FFmpeg:"
    echo "  macOS: brew install ffmpeg"
    echo "  Linux: sudo apt-get install ffmpeg"
fi

# Check .env file
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    exit 1
fi

echo "=========================================="
echo "Starting FastAPI server on port 8000..."
echo "=========================================="

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
