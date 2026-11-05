@echo off
setlocal
chcp 65001 >nul

rem ========================================
rem  CooLM Chat Launcher
rem  Resolves the project directory from this script's location, so it works
rem  on any machine and from any working directory (FIND-007).
rem ========================================

cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Could not enter the project directory: "%~dp0"
    pause
    exit /b 1
)

rem ---- Locate Python: prefer "python", fall back to the "py" launcher ----
set "PY=python"
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python was not found on this machine.
        echo Please install Python 3.10 or newer and make sure it is on PATH.
        pause
        exit /b 1
    )
    set "PY=py"
)

rem ---- Dependencies: only install when the imports are missing ----
%PY% -c "import torch, onnxruntime, tokenizers" >nul 2>&1
if errorlevel 1 (
    echo ========================================
    echo  Installing dependencies...
    echo ========================================
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies. See the messages above.
        pause
        exit /b 1
    )
)

echo ========================================
echo  Starting CooLM Local Web Server...
echo ========================================
%PY% -m tools.server
if errorlevel 1 (
    echo.
    echo ERROR: The CooLM server exited with error code %errorlevel%.
    pause
    exit /b 1
)
pause
