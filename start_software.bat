@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" gui.py
) else if exist "%LocalAppData%\Programs\Python\Python37\python.exe" (
    "%LocalAppData%\Programs\Python\Python37\python.exe" gui.py
) else (
    py -3.7 gui.py
)
if errorlevel 1 (
    echo Startup failed. Run setup_windows.bat first and review README.md.
    pause
    exit /b 1
)
