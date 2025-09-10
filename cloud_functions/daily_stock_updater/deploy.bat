@echo off
REM Deploy Daily Stock Updater Cloud Function (Windows version)
REM Usage: deploy.bat [your-alpha-vantage-api-key] [your-gcp-project-id]

setlocal enabledelayedexpansion

REM Configuration
set FUNCTION_NAME=daily-stock-update
set REGION=us-central1
set RUNTIME=python311
set MEMORY=512MB
set TIMEOUT=540s

REM Get parameters
if "%~1"=="" (
    set ALPHA_VANTAGE_API_KEY=%ALPHA_VANTAGE_API_KEY%
) else (
    set ALPHA_VANTAGE_API_KEY=%~1
)

if "%~2"=="" (
    set BIGQUERY_PROJECT_ID=%GOOGLE_CLOUD_PROJECT%
) else (
    set BIGQUERY_PROJECT_ID=%~2
)

if "%ALPHA_VANTAGE_API_KEY%"=="" (
    echo Error: ALPHA_VANTAGE_API_KEY is required
    echo Usage: %0 [alpha-vantage-key] [gcp-project-id]
    echo Or set environment variables ALPHA_VANTAGE_API_KEY and GOOGLE_CLOUD_PROJECT
    exit /b 1
)

if "%BIGQUERY_PROJECT_ID%"=="" (
    echo Error: BIGQUERY_PROJECT_ID is required
    echo Usage: %0 [alpha-vantage-key] [gcp-project-id]
    echo Or set environment variables ALPHA_VANTAGE_API_KEY and GOOGLE_CLOUD_PROJECT
    exit /b 1
)

echo Deploying Cloud Function: %FUNCTION_NAME%
echo Project: %BIGQUERY_PROJECT_ID%
echo Region: %REGION%

REM Deploy the function
gcloud functions deploy %FUNCTION_NAME% ^
  --runtime %RUNTIME% ^
  --trigger-http ^
  --allow-unauthenticated ^
  --region %REGION% ^
  --memory %MEMORY% ^
  --timeout %TIMEOUT% ^
  --set-env-vars "ALPHA_VANTAGE_API_KEY=%ALPHA_VANTAGE_API_KEY%,BIGQUERY_PROJECT_ID=%BIGQUERY_PROJECT_ID%,MAX_TRADING_DAYS=500" ^
  --source . ^
  --entry-point daily_stock_update

if %errorlevel% neq 0 (
    echo Function deployment failed!
    exit /b 1
)

echo Function deployed successfully!

REM Get the function URL
for /f "delims=" %%i in ('gcloud functions describe %FUNCTION_NAME% --region=%REGION% --format="value(httpsTrigger.url)"') do set FUNCTION_URL=%%i
echo Function URL: %FUNCTION_URL%

REM Ask about Cloud Scheduler job
set /p CREATE_SCHEDULER="Create a daily Cloud Scheduler job? (y/n): "
if /i "%CREATE_SCHEDULER%"=="y" (
    set SCHEDULER_JOB_NAME=daily-stock-update-job
    
    echo Creating Cloud Scheduler job: !SCHEDULER_JOB_NAME!
    
    gcloud scheduler jobs create http !SCHEDULER_JOB_NAME! ^
      --schedule="0 22 * * MON-FRI" ^
      --uri="!FUNCTION_URL!" ^
      --http-method=POST ^
      --time-zone="America/New_York" ^
      --description="Daily stock data update (weekdays at 10 PM ET) - Auto-discovers symbols from BigQuery" ^
      --headers="Content-Type=application/json" ^
      --message-body="{}"
    
    if !errorlevel! equ 0 (
        echo Scheduler job created successfully!
        echo Job will run weekdays at 10 PM ET (after market close)
    ) else (
        echo Warning: Scheduler job creation failed
    )
)

echo Deployment complete!
echo.
echo Next steps:
echo 1. Test the function: curl -X POST %FUNCTION_URL%
echo 2. Monitor logs: gcloud functions logs read %FUNCTION_NAME% --region=%REGION%
echo 3. Update your Streamlit app to use the fresh BigQuery data

endlocal
