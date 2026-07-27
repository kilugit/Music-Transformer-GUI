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

echo Installing PyTorch (CPU) and torch-directml...
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo PyTorch installation failed.
    pause
    exit /b %errorlevel%
)

pip install --no-cache-dir torch-directml
if %errorlevel% neq 0 (
    echo torch-directml installation failed.
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
