@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
if exist "%LocalAppData%\Programs\Python\Python37\python.exe" (
    "%LocalAppData%\Programs\Python\Python37\python.exe" -m venv .venv
) else (
    py -3.7 -m venv .venv
)
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 7) else 1)"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" download_resources.py
if errorlevel 1 goto failed
echo Setup complete. Double-click start_software.bat to open the application.
pause
exit /b 0
:failed
echo Setup failed. Python 3.7, Tcl/Tk, pip and network access are required.
echo Review the error above and the README.md instructions.
pause
exit /b 1
