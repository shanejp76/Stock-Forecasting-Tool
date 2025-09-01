# Stock Forecasting Tool Environment Activation Script
# Usage: .\activate_env.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Stock Forecasting Tool Environment" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Activating conda environment..." -ForegroundColor Yellow

# Check if conda is available
try {
    $condaVersion = conda --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Conda found: $condaVersion" -ForegroundColor Green
    } else {
        throw "Conda not found"
    }
} catch {
    Write-Host "❌ Conda is not available in PATH" -ForegroundColor Red
    Write-Host "Solutions:" -ForegroundColor Yellow
    Write-Host "1. Install Anaconda/Miniconda" -ForegroundColor White
    Write-Host "2. Add conda to PATH" -ForegroundColor White
    Write-Host "3. Use 'Anaconda Prompt' instead of PowerShell" -ForegroundColor White
    pause
    exit 1
}

# Activate environment
conda activate stock-forecasting

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Environment activated successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to activate environment 'stock-forecasting'" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check available environments: conda env list" -ForegroundColor White
    Write-Host "2. Create environment: conda create -n stock-forecasting python=3.11" -ForegroundColor White
    Write-Host "3. Install requirements: pip install -r requirements.txt" -ForegroundColor White
    pause
    exit 1
}

Write-Host ""
Write-Host "Current Python setup:" -ForegroundColor Cyan
python --version
Write-Host "Python location:" -ForegroundColor White
python -c "import sys; print(sys.executable)"
Write-Host ""

Write-Host "Checking required packages..." -ForegroundColor Cyan
python -c "
try:
    import streamlit
    print('✅ Streamlit:', streamlit.__version__)
except ImportError:
    print('❌ Streamlit not installed - run: pip install streamlit')

try:
    import google.cloud.bigquery
    print('✅ BigQuery client available')
except ImportError:
    print('❌ BigQuery client not available - run: pip install google-cloud-bigquery')

try:
    from app_modules.bigquery_client import BigQueryClient
    print('✅ Custom BigQuery client available')
except ImportError as e:
    print('❌ Custom BigQuery client error:', str(e))

import os
if os.path.exists('.env'):
    print('✅ Environment file (.env) found')
else:
    print('❌ Environment file (.env) missing')
"

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Ready for development!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Quick commands:" -ForegroundColor Yellow
Write-Host "  python -m streamlit run main.py                    # Launch Streamlit app" -ForegroundColor White
Write-Host "  python scripts/initial_bulk_load.py --help         # Bulk loading options" -ForegroundColor White
Write-Host "  python -c `"from app_modules.bigquery_client import BigQueryClient; print(f'Symbols: {len(BigQueryClient().get_available_symbols())}')`"  # Check data" -ForegroundColor White
Write-Host ""
