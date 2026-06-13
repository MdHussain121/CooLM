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
echo  CooLM Pipeline - Step 1: Generate Data
echo ========================================
python generate_data.py 500000
if %errorlevel% neq 0 (
    echo FAILED at Step 1
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo  CooLM Pipeline - Step 2: Prepare Data
echo ========================================
python -m coolm.prepare_data pigeon_data.jsonl
if %errorlevel% neq 0 (
    echo FAILED at Step 2
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo  CooLM Pipeline - Step 3: Train Model
echo ========================================
python -m coolm.train
if %errorlevel% neq 0 (
    echo FAILED at Step 3
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo  Pipeline Complete!
echo.
echo  Chat with Coo: chat.bat
echo ========================================
pause
