@echo off
REM Quick Start Script for Recursive Code LLM - Windows Version
REM This script helps you get started quickly on Windows

echo ==================================
echo Recursive Code LLM - Quick Start
echo ==================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.9+ from python.org
    pause
    exit /b 1
)
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate venv
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Some packages failed to install
    echo You may need to install CUDA toolkit for GPU support
)
echo [OK] Dependencies installed

REM Create necessary directories
echo.
echo Creating directories...
if not exist "checkpoints\" mkdir checkpoints
if not exist "logs\" mkdir logs
if not exist "data\" mkdir data
echo [OK] Directories created

REM Check CUDA availability
echo.
echo Checking CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
echo ==================================
echo Setup complete!
echo ==================================
echo.
echo To start training, run:
echo   python train.py --config config/training_config.yaml
echo.
echo To resume from checkpoint:
echo   python train.py --config config/training_config.yaml --resume
echo.
echo To run tests:
echo   pytest tests/
echo.
echo NOTE: Keep this terminal window open to use the virtual environment
echo       Or activate it manually with: venv\Scripts\activate.bat
echo.
pause
