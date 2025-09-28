@echo off
setlocal enabledelayedexpansion

REM Deploy Parameter Optimizer Cloud Function (Windows version)
REM Usage: deploy.bat [your-gcp-project-id]
REM
REM This function runs weekly parameter optimization after raw data updates complete
REM on the last trading day of each week.

echo ======================================
echo Parameter Optimizer Deployment Script
echo ======================================

REM Check if gcloud is installed
gcloud version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Google Cloud CLI is not installed or not in PATH
    echo Please install from: https://cloud.google.com/sdk/docs/install
    exit /b 1
)

REM Get project ID from command line or prompt
set PROJECT_ID=%1
if "%PROJECT_ID%"=="" (
    set /p PROJECT_ID="Enter your GCP Project ID: "
)

if "%PROJECT_ID%"=="" (
    echo Error: Project ID is required
    exit /b 1
)

REM Set configuration variables
set FUNCTION_NAME=parameter-optimizer
set REGION=us-central1
set MEMORY=2Gi
set TIMEOUT=3600
set DATASET_ID=stock_data
set RAW_DATA_TABLE=raw_stock_data
set OPTIMAL_PARAMS_TABLE=optimal_parameters

echo.
echo Configuration:
echo - Function Name: %FUNCTION_NAME%
echo - Project ID: %PROJECT_ID%
echo - Region: %REGION%
echo - Memory: %MEMORY%
echo - Timeout: %TIMEOUT%s
echo - Dataset: %DATASET_ID%
echo - Raw Data Table: %RAW_DATA_TABLE%
echo - Optimal Parameters Table: %OPTIMAL_PARAMS_TABLE%
echo.

REM Set the project
echo Setting GCP project to %PROJECT_ID%...
gcloud config set project %PROJECT_ID%

REM Deploy the function
echo Deploying Cloud Function: %FUNCTION_NAME%
echo This may take a few minutes...
echo.

gcloud functions deploy %FUNCTION_NAME% ^
  --gen2 ^
  --runtime=python311 ^
  --region=%REGION% ^
  --source=. ^
  --entry-point=parameter_optimization ^
  --trigger=http ^
  --memory=%MEMORY% ^
  --timeout=%TIMEOUT% ^
  --set-env-vars=BIGQUERY_PROJECT_ID=%PROJECT_ID%,BIGQUERY_DATASET_ID=%DATASET_ID%,RAW_DATA_TABLE_ID=%RAW_DATA_TABLE%,OPTIMAL_PARAMS_TABLE_ID=%OPTIMAL_PARAMS_TABLE% ^
  --no-allow-unauthenticated

if %errorlevel% neq 0 (
    echo Function deployment failed!
    exit /b 1
)

echo Function deployed successfully!

REM Get function URL
echo Getting function URL...
for /f "tokens=*" %%i in ('gcloud functions describe %FUNCTION_NAME% --region=%REGION% --format="value(serviceConfig.uri)"') do set FUNCTION_URL=%%i

if "%FUNCTION_URL%"=="" (
    echo Warning: Could not retrieve function URL
    set FUNCTION_URL=https://%REGION%-%PROJECT_ID%.cloudfunctions.net/%FUNCTION_NAME%
)

echo Function URL: %FUNCTION_URL%

REM Ask about Cloud Scheduler job
set /p CREATE_SCHEDULER="Create a weekly Cloud Scheduler job for parameter optimization? (y/n): "
if /i "%CREATE_SCHEDULER%"=="y" (
    set SCHEDULER_JOB_NAME=weekly-parameter-optimization-job
    
    echo Creating Cloud Scheduler job: !SCHEDULER_JOB_NAME!
    echo.
    echo Scheduling Options:
    echo 1. Conservative: Fridays at 11:30 PM ET (30 min after raw data)
    echo 2. Aggressive: Fridays at 11:00 PM ET (same time as raw data, uses waiting logic)
    echo 3. Safe: Saturdays at 12:30 AM ET (1.5 hours after raw data, guaranteed fresh)
    echo.
    set /p SCHEDULE_CHOICE="Choose scheduling option (1/2/3): "
    
    if "!SCHEDULE_CHOICE!"=="1" (
        set CRON_SCHEDULE=30 23 * * FRI
        set SCHEDULE_DESC=Fridays at 11:30 PM ET - Conservative approach
        set MESSAGE_BODY={\"force_run\": false}
    ) else if "!SCHEDULE_CHOICE!"=="2" (
        set CRON_SCHEDULE=0 23 * * FRI
        set SCHEDULE_DESC=Fridays at 11:00 PM ET - Uses waiting logic
        set MESSAGE_BODY={\"force_run\": false}
    ) else (
        set CRON_SCHEDULE=30 0 * * SAT
        set SCHEDULE_DESC=Saturdays at 12:30 AM ET - Safe approach
        set MESSAGE_BODY={\"force_run\": true}
    )
    
    echo Creating job with schedule: !CRON_SCHEDULE!
    echo Description: !SCHEDULE_DESC!
    
    gcloud scheduler jobs create http !SCHEDULER_JOB_NAME! ^
      --schedule="!CRON_SCHEDULE!" ^
      --uri="!FUNCTION_URL!" ^
      --http-method=POST ^
      --time-zone="America/New_York" ^
      --description="Weekly parameter optimization - !SCHEDULE_DESC!" ^
      --headers="Content-Type=application/json" ^
      --message-body="!MESSAGE_BODY!"
    
    if !errorlevel! equ 0 (
        echo Scheduler job created successfully!
        echo Schedule: !SCHEDULE_DESC!
        echo This ensures optimal timing relative to raw data updates
        echo.
        echo Optional: Create a backup Saturday job? (y/n^)
        set /p CREATE_BACKUP=""
        if /i "!CREATE_BACKUP!"=="y" (
            set BACKUP_JOB_NAME=weekly-parameter-optimization-backup
            echo Creating backup job for Saturday mornings...
            gcloud scheduler jobs create http !BACKUP_JOB_NAME! ^
              --schedule="0 6 * * SAT" ^
              --uri="!FUNCTION_URL!" ^
              --http-method=POST ^
              --time-zone="America/New_York" ^
              --description="Weekly parameter optimization backup (Saturday 6 AM ET)" ^
              --headers="Content-Type=application/json" ^
              --message-body="{\"force_run\": true}"
            
            if !errorlevel! equ 0 (
                echo Backup scheduler job created successfully!
            ) else (
                echo Warning: Backup scheduler job creation failed
            )
        )
    ) else (
        echo Warning: Scheduler job creation failed
        echo You can create it manually later if needed
    )
)

echo.
echo =======================================
echo Deployment Complete!
echo =======================================
echo.
echo Function Details:
echo - Name: %FUNCTION_NAME%
echo - URL: %FUNCTION_URL%
echo - Memory: %MEMORY%
echo - Timeout: %TIMEOUT%s
echo.
echo Next steps:
echo 1. Test the function manually (requires authentication):
echo    For Windows PowerShell:
echo      $token = gcloud auth print-identity-token
echo      $headers = @{"Authorization"="Bearer $token"}
echo      Invoke-WebRequest -Uri "%FUNCTION_URL%" -Method POST -Headers $headers -Body '{"force_run": true}' -ContentType "application/json"
echo.
echo    For Command Line:
echo      curl -X POST "%FUNCTION_URL%" -H "Authorization: Bearer $(gcloud auth print-identity-token)" -H "Content-Type: application/json" -d "{\"force_run\": true}"
echo.
echo 2. Monitor logs:
echo    gcloud functions logs read %FUNCTION_NAME% --region=%REGION%
echo.
echo 3. The function will automatically run weekly after raw data updates
echo    or can be triggered manually with the curl command above
echo.
echo 4. Check Cloud Scheduler:
echo    gcloud scheduler jobs list

endlocal