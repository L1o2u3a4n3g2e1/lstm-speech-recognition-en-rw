@echo off
echo.
echo ================================================================
echo   LSTM Kinyarwanda Speech Recognition - Training Manager
echo ================================================================
echo.

setlocal enabledelayedexpansion

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo.
echo Menu:
echo --------
echo 1. Generate Synthetic Audio (2000+ samples from corpus)
echo 2. Continue Training Model (from synthetic + original data)
echo 3. Run Web App (with training interface)
echo 4. Check Training Status
echo 5. View Training Logs
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Generating synthetic audio from Kinyarwanda-English corpus...
    echo This may take 15-30 minutes depending on your system...
    echo.
    python scripts\generate_synthetic_audio.py --num-samples 2000 --combine
    echo.
    echo Synthetic audio generation complete!
    echo Next: Run option 2 to train the model on the new data.
    pause

) else if "%choice%"=="2" (
    echo.
    echo Starting model training...
    echo Check train_masked.log for progress.
    echo This will run in the background and take 1-2 hours.
    echo.
    echo Note: You can open another terminal and run the web app (option 3)
    echo while training is happening in the background.
    echo.
    set /p num_epochs="How many epochs to train? (default 100): "
    if "!num_epochs!"=="" set num_epochs=100

    python scripts\train_with_masking.py > train_masked.log 2>&1
    echo.
    echo Training started!
    echo Logs saved to: train_masked.log
    pause

) else if "%choice%"=="3" (
    echo.
    echo Starting Web Application...
    echo.
    cd web_app

    REM Check if virtual environment exists
    if not exist "venv" (
        echo Creating virtual environment...
        python -m venv venv
        call venv\Scripts\activate.bat
        echo Installing dependencies...
        pip install -r requirements.txt
    ) else (
        call venv\Scripts\activate.bat
    )

    echo.
    echo ================================================================
    echo   Web App Starting...
    echo   Open your browser to: http://localhost:5000
    echo
    echo   Features:
    echo   - Transcribe Kinyarwanda speech
    echo   - Train model on your own samples
    echo   - View transcription history
    echo   - Access on mobile via: http://^<your-ip^>:5000
    echo ================================================================
    echo.

    python app.py

) else if "%choice%"=="4" (
    echo.
    echo Training Status:
    echo --------
    if exist "train_masked.log" (
        echo Latest 20 lines from training log:
        echo.
        powershell -Command "Get-Content -Path train_masked.log -Tail 20"
    ) else (
        echo No training log found yet.
        echo Run option 2 to start training.
    )
    echo.
    pause

) else if "%choice%"=="5" (
    echo.
    echo Opening training logs...
    echo.
    if exist "train_masked.log" (
        echo Showing full training log:
        powershell -Command "Get-Content -Path train_masked.log | Less"
    ) else (
        echo No training log found.
    )
    pause

) else (
    echo Invalid choice. Please run the script again.
    pause
    exit /b 1
)

pause
