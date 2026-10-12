@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python launcher not found. Install Python 3.12 from python.org and enable the launcher.
    pause
    exit /b 1
  )
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Python 3.12 is required by this launcher. See README for Python 3.11 setup.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -c "import numpy, PIL" >nul 2>nul
if errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection and README.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe run.py
pause
