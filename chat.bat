@echo off
cd /d "C:\Users\moham\Music\datamethod1"

echo ========================================
echo  Installing Dependencies...
echo ========================================
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b %errorlevel%
)

echo ========================================
echo  Starting CooLM Local Web Server...
echo ========================================
python -m tools.server
if %errorlevel% neq 0 (
    echo Server ended with an error.
    pause
    exit /b %errorlevel%
)
pause
