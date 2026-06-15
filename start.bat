@echo off
REM ===========================================================================
REM  BroN-translate (Local Edition) - Windows launcher
REM  Double-click this file. First run sets up a venv and installs deps;
REM  later runs just start the server and open your browser.
REM ===========================================================================
setlocal
cd /d "%~dp0"

REM --- find Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.10+ from python.org
    echo         and make sure "Add Python to PATH" is checked.
    pause
    exit /b 1
)

REM --- create venv on first run ---
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment ^(first run only^)...
    python -m venv .venv
    if errorlevel 1 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )
    echo [*] Installing dependencies... this can take a while.
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 ( echo [ERROR] dependency install failed. & pause & exit /b 1 )
)

REM --- first-run .env reminder ---
if not exist ".env" (
    echo [!] No .env found. Copying .env.example -> .env
    copy /y ".env.example" ".env" >nul
    echo [!] Open .env and paste your DEEPSEEK_API_KEY, then run this again.
    notepad ".env"
    pause
    exit /b 0
)

echo [*] Starting BroN-translate (Local)...
".venv\Scripts\python.exe" main.py
pause
endlocal
