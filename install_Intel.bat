@echo off
chcp 65001 >nul
echo Checking for Python 3.12...
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.12 not found. Downloading...
    curl -sL -o "%TEMP%\python-3.12.9-amd64.exe" https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe
    if %errorlevel% neq 0 (
        echo Failed to download Python 3.12.
        pause
        exit /b %errorlevel%
    )
    echo Installing Python 3.12...
    start /wait "" "%TEMP%\python-3.12.9-amd64.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    if %errorlevel% neq 0 (
        echo Failed to install Python 3.12.
        pause
        exit /b %errorlevel%
    )
    del "%TEMP%\python-3.12.9-amd64.exe"
    echo Python 3.12 installed successfully.
)
echo Creating virtual environment with Python 3.12...
py -3.12 -m venv venv
if %errorlevel% neq 0 (
    echo Failed to create venv. Make sure Python 3.12 is installed.
    pause
    exit /b %errorlevel%
)

call venv\Scripts\activate

echo Installing PyTorch for Intel Arc GPUs (XPU)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
if %errorlevel% neq 0 (
    echo PyTorch XPU installation failed.
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
