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

REM Install PyTorch first (required for other packages)
echo.
echo Installing PyTorch with CUDA 12.x support (this may take a few minutes)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo WARNING: Failed to install PyTorch with CUDA 12.x
    echo Trying CUDA 11.8 version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if %errorlevel% neq 0 (
        echo WARNING: Failed with CUDA 11.8, trying CPU-only version...
        pip install torch torchvision torchaudio
    )
)

REM Check if PyTorch installed successfully
python -c "import torch" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: PyTorch installation failed. Please install manually.
    echo Visit: https://pytorch.org/get-started/locally/
    pause
    exit /b 1
)
echo [OK] PyTorch installed

REM Install other dependencies
echo.
echo Installing other dependencies...
echo This may take a few minutes...
pip install -r requirements.txt --no-deps
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Some optional packages may have failed
    echo The core functionality should still work
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
echo ==========================================
echo Checking GPU/CUDA availability...
echo ==========================================
python -c "import torch; cuda_available = torch.cuda.is_available(); print(f'\nPyTorch version: {torch.__version__}'); print(f'CUDA available: {cuda_available}'); print(f'CUDA version: {torch.version.cuda if cuda_available else \"N/A\"}'); print(f'GPU device: {torch.cuda.get_device_name(0) if cuda_available else \"CPU only\"}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if cuda_available else '')"
if %errorlevel% neq 0 (
    echo WARNING: Could not check CUDA status
)

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Quick commands:
echo.
echo   Start training:
echo     python train.py --config config\training_config.yaml
echo.
echo   Resume from checkpoint:
echo     python train.py --resume
echo.
echo   Generate code (after training):
echo     python inference.py --model checkpoints\final_model\pytorch_model.bin --tokenizer checkpoints\tokenizer.json
echo.
echo   Run tests:
echo     pytest tests\
echo.
echo NOTE: This terminal has the virtual environment activated.
echo       Next time, activate it with: venv\Scripts\activate.bat
echo.
echo If CUDA is not available but you have an NVIDIA GPU:
echo   1. Verify CUDA Toolkit is installed: nvidia-smi
echo   2. Reinstall PyTorch for your CUDA version:
echo      CUDA 12.x: pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu121
echo      CUDA 11.8: pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu118
echo.
pause
