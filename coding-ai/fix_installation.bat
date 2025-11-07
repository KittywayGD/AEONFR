@echo off
REM Fix script for installation issues
REM Run this if quick_start.bat had errors

echo ==========================================
echo Installation Fix Script
echo ==========================================
echo.
echo This script will:
echo 1. Clean up failed installations
echo 2. Install PyTorch first (with CUDA)
echo 3. Install remaining dependencies
echo.
pause

REM Activate venv if it exists
if exist "venv\" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please run quick_start.bat first
    pause
    exit /b 1
)

REM Uninstall problematic packages
echo.
echo Cleaning up...
pip uninstall -y deepspeed torch 2>nul

REM Install PyTorch with CUDA support
echo.
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Check installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: PyTorch installation failed
    echo Trying CPU-only version...
    pip install torch torchvision torchaudio
)

REM Install remaining dependencies
echo.
echo Installing remaining dependencies...
pip install -r requirements.txt

echo.
echo ==========================================
echo Fix complete!
echo ==========================================
echo.
echo Check if CUDA is working:
python -c "import torch; print(f'\nCUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

echo.
echo If CUDA is still not available:
echo   1. Verify your NVIDIA drivers are up to date
echo   2. Install CUDA Toolkit 11.8 from nvidia.com/cuda
echo   3. Restart your computer
echo   4. Run this script again
echo.
pause
