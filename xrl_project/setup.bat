@echo off
REM =========================================================================
REM X-Ray Lithography Project Setup Script (Windows)
REM Author: Abhineet Agarwal
REM Course: ME6110
REM Date: November 2025
REM =========================================================================

echo ========================================================================
echo X-RAY LITHOGRAPHY PROJECT - SETUP (WINDOWS)
echo ========================================================================
echo.

REM Check Python installation
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

python --version
echo.

REM Install required packages
echo Installing required Python packages...
echo This may take a few minutes...
echo.

python -m pip install --user numpy scipy matplotlib gdspy

if errorlevel 1 (
    echo.
    echo WARNING: Some packages may have failed to install
    echo Try running: python -m pip install --user numpy scipy matplotlib gdspy
) else (
    echo.
    echo All packages installed successfully!
)

echo.
echo ========================================================================
echo SETUP COMPLETE
echo ========================================================================
echo.
echo Quick Start:
echo.
echo 1. Run all simulations:
echo    cd simulations
echo    python run_all_simulations.py
echo.
echo 2. Generate GDS layouts:
echo    cd layouts
echo    python generate_layouts.py
echo.
echo 3. View results:
echo    - Plots: data\*.png
echo    - Summary: data\simulation_summary.txt
echo    - GDS file: layouts\xrl_test_patterns.gds
echo.
echo Documentation:
echo    - README.md - Complete project guide
echo    - QUICK_START.md - Quick reference
echo    - docs\beamtime_proposal.md - Experimental plan
echo.
echo ========================================================================
echo.
pause
