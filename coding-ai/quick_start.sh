#!/bin/bash

# Quick Start Script for Recursive Code LLM
# This script helps you get started quickly with the project

set -e  # Exit on error

echo "=================================="
echo "Recursive Code LLM - Quick Start"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate venv
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p checkpoints logs data
echo "✓ Directories created"

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "To start training, run:"
echo "  python train.py --config config/training_config.yaml"
echo ""
echo "To resume from checkpoint:"
echo "  python train.py --config config/training_config.yaml --resume"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
