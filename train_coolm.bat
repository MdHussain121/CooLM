@echo off
setlocal
chcp 65001 >nul

rem ========================================
rem  CooLM Full Training Pipeline
rem  generate data -> prepare -> train -> export ONNX (FIND-026)
rem  Resolves the project directory from this script's location (FIND-007).
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

echo.
echo ========================================
echo  CooLM Pipeline - Step 1/4: Generate Data
echo ========================================
%PY% generate_data.py 500000
if errorlevel 1 (
    echo FAILED at Step 1 - generate_data. See the messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  CooLM Pipeline - Step 2/4: Prepare Data
echo ========================================
%PY% -m coolm.prepare_data pigeon_data.jsonl
if errorlevel 1 (
    echo FAILED at Step 2 - prepare_data. See the messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  CooLM Pipeline - Step 3/4: Train Model
echo ========================================
%PY% -m coolm.train
if errorlevel 1 (
    echo FAILED at Step 3 - train. See the messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  CooLM Pipeline - Step 4/4: Export ONNX
echo ========================================
%PY% -m tools.model_export
if errorlevel 1 (
    echo FAILED at Step 4 - ONNX export. See the messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Pipeline Complete!
echo.
echo  The model is now chat-ready. Artifacts produced:
echo    checkpoints\best_model.pt   (PyTorch checkpoint)
echo    docs\model.onnx             (ONNX export used by the chat server)
echo    data\tokenizer.json         (tokenizer used by the chat server)
echo.
echo  Chat with Coo now: chat.bat
echo ========================================
pause
