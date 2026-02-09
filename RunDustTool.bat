@echo off
REM Go to the folder where this script lives
cd /d "%~dp0"

REM Run the Python script (adjust 'python' to 'py' if needed)
python dust_analysis.py

echo.
echo Done. Outputs are in the "results" folder.
echo Press any key to close this window...
pause >nul
