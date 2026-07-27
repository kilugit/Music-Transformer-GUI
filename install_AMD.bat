@echo off
chcp 65001 >nul
echo Creating virtual environment with Python 3.12...
py -3.12 -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create venv. Make sure Python 3.12 is installed.
    pause
    exit /b %errorlevel%
)

call venv\Scripts\activate

echo Installing ROCm packages...
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
if %errorlevel% neq 0 (
    echo ROCm SDK installation failed.
    pause
    exit /b %errorlevel%
)

pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%%2Brocm7.2.1-cp312-cp312-win_amd64.whl
if %errorlevel% neq 0 (
    echo PyTorch ROCm installation failed.
    pause
    exit /b %errorlevel%
)

echo Installing remaining dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo pip install failed.
    pause
    exit /b %errorlevel%
)

echo.
echo Installation complete! Run run.bat to launch the GUI.
pause
