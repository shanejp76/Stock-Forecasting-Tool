@echo off
REM Stock Forecasting Tool - Environment Activation Script (Batch)
REM This script activates the development environment and checks all dependencies

echo.
echo =============================================================
echo  Stock Forecasting Tool - Environment Activation
echo =============================================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% == 0 (
    echo [INFO] Conda detected - attempting to activate environment...
    call conda activate stock-forecasting 2>nul
    if %errorlevel% == 0 (
        echo [SUCCESS] Activated conda environment: stock-forecasting
    ) else (
        echo [WARNING] Conda environment 'stock-forecasting' not found
        echo [INFO] You may need to create it with: conda create -n stock-forecasting python=3.11
    )
) else (
    echo [INFO] Conda not found - using system Python
)

echo.
echo [INFO] Checking Python environment...
python --version
python -c "import sys; print('Python executable:', sys.executable)"

echo.
echo [INFO] Installing/updating required packages...
python -m pip install --upgrade pip
python -m pip install python-dotenv

echo.
echo [INFO] Running comprehensive environment check...
python debug_scripts/check_env.py

echo.
if %errorlevel% == 0 (
    echo [SUCCESS] Ready for development!
    echo.
    echo Available commands:
    echo   python -m streamlit run main.py          - Launch Streamlit application
    echo   python scripts/initial_bulk_load.py      - Bulk data loading
    echo   python debug_scripts/check_env.py        - Environment verification
) else (
    echo [WARNING] Some issues detected. Please review the output above.
)

echo.
echo Press any key to continue...
pause >nul
