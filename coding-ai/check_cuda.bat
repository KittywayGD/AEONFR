@echo off
REM Quick CUDA and PyTorch verification script
REM Run this to check if CUDA is properly configured

echo ==========================================
echo CUDA and PyTorch Configuration Check
echo ==========================================
echo.

REM Check if virtual environment exists and activate
if exist "venv\" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [WARNING] Virtual environment not found
    echo Please run quick_start.bat first
    echo.
    pause
    exit /b 0
)

echo.
echo Checking NVIDIA driver and CUDA...
echo ==========================================
nvidia-smi
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] nvidia-smi not found!
    echo Make sure you have NVIDIA drivers installed.
    echo.
    pause
    exit /b 1
)

echo.
echo.
echo Checking PyTorch installation...
echo ==========================================
python -c "import sys; import torch; print(f'Python version: {sys.version.split()[0]}'); print(f'PyTorch version: {torch.__version__}'); cuda_available = torch.cuda.is_available(); print(f'\nCUDA available: {cuda_available}'); print(f'CUDA version (PyTorch): {torch.version.cuda if cuda_available else \"N/A\"}'); print(f'cuDNN version: {torch.backends.cudnn.version() if cuda_available else \"N/A\"}'); print(f'\nGPU detected: {torch.cuda.get_device_name(0) if cuda_available else \"None\"}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if cuda_available else '')"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyTorch is not installed!
    echo Run quick_start.bat or fix_installation.bat
    echo.
    pause
    exit /b 1
)

echo.
echo.
echo Quick GPU Test...
echo ==========================================
python -c "import torch; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); x = torch.rand(1000, 1000, device=device); y = torch.rand(1000, 1000, device=device); z = torch.matmul(x, y); print(f'Matrix multiplication test: SUCCESS'); print(f'Computation device: {device}'); print(f'Result shape: {z.shape}')"
if %errorlevel% neq 0 (
    echo [WARNING] GPU test failed
) else (
    echo [OK] GPU is working correctly!
)

echo.
echo ==========================================
echo Configuration Summary
echo ==========================================
python -c "import torch; cuda_ok = torch.cuda.is_available(); print(f'\n Status: {\"READY FOR TRAINING\" if cuda_ok else \"CPU ONLY MODE\"}'); print(f'\n GPU Training: {\"YES\" if cuda_ok else \"NO\"}'); print(f' Mixed Precision: {\"YES\" if cuda_ok else \"NO (CPU mode)\"}'); print(f' Recommended batch_size: {\"2-4\" if cuda_ok else \"1\"}')"

echo.
echo ==========================================
echo.
echo If CUDA is not available:
echo   1. Verify CUDA Toolkit is installed
echo   2. Check nvidia-smi shows your GPU
echo   3. Reinstall PyTorch:
echo      CUDA 12.x: pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu121
echo      CUDA 11.8: pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu118
echo   4. Run this script again to verify
echo.
pause
