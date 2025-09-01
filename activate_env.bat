@echo off
echo ================================
echo Stock Forecasting Tool Environment
echo ================================
echo.

echo Activating conda environment...
call conda activate stock-forecasting

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to activate conda environment 'stock-forecasting'
    echo.
    echo Troubleshooting:
    echo 1. Check if conda is installed: conda --version
    echo 2. Check available environments: conda env list
    echo 3. Create environment if missing: conda create -n stock-forecasting python=3.11
    echo.
    pause
    exit /b 1
)

echo.
echo Environment activated successfully!
echo.

echo Current Python setup:
python --version
echo Python location: 
python -c "import sys; print(sys.executable)"
echo.

echo Checking required packages...
python -c "
try:
    import streamlit
    print('✅ Streamlit:', streamlit.__version__)
except ImportError:
    print('❌ Streamlit not installed')

try:
    import google.cloud.bigquery
    print('✅ BigQuery client available')
except ImportError:
    print('❌ BigQuery client not available')

try:
    from app_modules.bigquery_client import BigQueryClient
    print('✅ Custom BigQuery client available')
except ImportError:
    print('❌ Custom BigQuery client not available')

import os
if os.path.exists('.env'):
    print('✅ Environment file (.env) found')
else:
    print('❌ Environment file (.env) missing')
"

echo.
echo ================================
echo Ready for development!
echo ================================
echo.
echo Available commands:
echo   python -m streamlit run main.py    - Launch Streamlit app
echo   python scripts/initial_bulk_load.py --help  - Bulk loading options
echo   python -c "from app_modules.bigquery_client import BigQueryClient; print(f'Symbols: {len(BigQueryClient().get_available_symbols())}')"  - Check data
echo.

cmd /k
