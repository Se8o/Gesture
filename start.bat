@echo off
setlocal

:: ─── Gesture Remote Controller – Windows launcher ───────────────────────────
:: Double-click this file to install and start the application.
:: Works without an IDE or terminal knowledge.
:: ─────────────────────────────────────────────────────────────────────────────

title Gesture Remote Controller

:: Move to the folder where this .bat file lives (project root)
cd /d "%~dp0"

echo ============================================
echo  Gesture Remote Controller
echo ============================================
echo.

:: ── 1. Check Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Please install Python 3.9-3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo Detected: && python --version
echo NOTE: mediapipe requires Python 3.8-3.11. Python 3.12+ is NOT supported.
echo.

:: ── 2. Create virtual environment if it does not exist ───────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Done.
) else (
    echo [1/4] Virtual environment already exists.
)

:: ── 3. Activate virtual environment ──────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── 4. Install / update dependencies ─────────────────────────────────────────
echo [2/4] Installing dependencies (this may take a minute on first run)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies. See the error above.
    echo Common fixes:
    echo   - Use Python 3.11 (mediapipe does not support 3.12+^)
    echo   - Check your internet connection
    echo   - Run as Administrator if you see permission errors
    pause
    exit /b 1
)
echo       Done.

:: ── 5. Train model if not present ────────────────────────────────────────────
if not exist "models\model.pkl" (
    echo [3/4] Model not found – training now (this takes ~30 seconds)...
    python ml\train.py
    if errorlevel 1 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
    echo       Done.
) else (
    echo [3/4] Model already trained.
)

:: ── 6. Start the application ──────────────────────────────────────────────────
echo [4/4] Starting Gesture Remote Controller...
echo.
echo Hold your hand in front of the camera and perform gestures.
echo Press Q in the camera window to quit.
echo.

python run.py

:: Keep window open if the app crashes so the user can read the error
if errorlevel 1 (
    echo.
    echo [ERROR] The application exited with an error. See message above.
    pause
)

endlocal
