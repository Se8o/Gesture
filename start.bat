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
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
echo Detected Python version:
python --version
echo NOTE: mediapipe requires Python 3.8-3.11. Python 3.12 and 3.13 are NOT supported.
echo If your version is 3.12 or higher, install Python 3.11 instead.
echo.

:: ── 2. Create virtual environment if it does not exist ───────────────────────
:: If a previous install failed the venv may be broken – delete and retry.
if exist "venv\Scripts\activate.bat" (
    echo [1/4] Virtual environment already exists.
) else (
    :: Clean up any leftover broken venv folder before creating a fresh one
    if exist "venv" (
        echo [1/4] Removing broken virtual environment...
        rmdir /s /q venv
    )
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo       Done.
)

:: ── 3. Activate virtual environment ──────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── 4. Install / update dependencies ─────────────────────────────────────────
echo [2/4] Installing dependencies (this may take a few minutes on first run)...
echo.

:: Upgrade pip first to avoid install failures caused by an outdated pip
python -m pip install --upgrade pip --quiet

pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  [ERROR] Dependency installation failed.
    echo  Read the red error lines above to find the cause.
    echo.
    echo  Most common fixes:
    echo    1. Wrong Python version - mediapipe needs Python 3.11 or
    echo       older. Run: python --version
    echo       If it says 3.12 or 3.13, install Python 3.11 instead.
    echo    2. No internet - check your network connection.
    echo    3. Antivirus / firewall blocking pip - try disabling it.
    echo    4. Try running this file as Administrator.
    echo ============================================================
    echo.
    :: Delete the broken venv so next run starts fresh
    echo Cleaning up broken virtual environment for next attempt...
    call venv\Scripts\deactivate.bat >nul 2>&1
    rmdir /s /q venv
    pause
    exit /b 1
)
echo       Done.

:: ── 5. Train model if not present ────────────────────────────────────────────
if not exist "models\model.pkl" (
    echo [3/4] Model not found - training now (this takes ~30 seconds^)...
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
