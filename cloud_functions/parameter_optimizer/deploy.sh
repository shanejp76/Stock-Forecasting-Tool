#!/bin/bash

# Deploy Parameter Optimizer Cloud Function
# Usage: ./deploy.sh [your-gcp-project-id]
#
# This function runs weekly parameter optimization after raw data updates complete
# on the last trading day of each week.

echo "======================================"
echo "Parameter Optimizer Deployment Script"
echo "======================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: Google Cloud CLI is not installed"
    echo "Please install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID from command line or prompt
PROJECT_ID=$1
if [ -z "$PROJECT_ID" ]; then
    read -p "Enter your GCP Project ID: " PROJECT_ID
fi

if [ -z "$PROJECT_ID" ]; then
    echo "Error: Project ID is required"
    exit 1
fi

# Set configuration variables
FUNCTION_NAME="parameter-optimizer"
REGION="us-central1"
MEMORY="2Gi"
TIMEOUT="3600"
DATASET_ID="stock_data"
RAW_DATA_TABLE="raw_stock_data"
OPTIMAL_PARAMS_TABLE="optimal_parameters"

echo
echo "Configuration:"
echo "- Function Name: $FUNCTION_NAME"
echo "- Project ID: $PROJECT_ID"
echo "- Region: $REGION"
echo "- Memory: $MEMORY"
echo "- Timeout: ${TIMEOUT}s"
echo "- Dataset: $DATASET_ID"
echo "- Raw Data Table: $RAW_DATA_TABLE"
echo "- Optimal Parameters Table: $OPTIMAL_PARAMS_TABLE"
echo

# Set the project
echo "Setting GCP project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Deploy the function
echo "Deploying Cloud Function: $FUNCTION_NAME"
echo "This may take a few minutes..."
echo

gcloud functions deploy $FUNCTION_NAME \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=parameter_optimization \
  --trigger=http \
  --memory=$MEMORY \
  --timeout=$TIMEOUT \
  --set-env-vars=BIGQUERY_PROJECT_ID=$PROJECT_ID,BIGQUERY_DATASET_ID=$DATASET_ID,RAW_DATA_TABLE_ID=$RAW_DATA_TABLE,OPTIMAL_PARAMS_TABLE_ID=$OPTIMAL_PARAMS_TABLE \
  --allow-unauthenticated

if [ $? -ne 0 ]; then
    echo "Function deployment failed!"
    exit 1
fi

echo "Function deployed successfully!"

# Get function URL
echo "Getting function URL..."
FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME --region=$REGION --format="value(serviceConfig.uri)")

if [ -z "$FUNCTION_URL" ]; then
    echo "Warning: Could not retrieve function URL"
    FUNCTION_URL="https://$REGION-$PROJECT_ID.cloudfunctions.net/$FUNCTION_NAME"
fi

echo "Function URL: $FUNCTION_URL"

# Ask about Cloud Scheduler job
read -p "Create a weekly Cloud Scheduler job for parameter optimization? (y/n): " CREATE_SCHEDULER
if [[ $CREATE_SCHEDULER =~ ^[Yy]$ ]]; then
    SCHEDULER_JOB_NAME="weekly-parameter-optimization-job"
    
    echo "Creating Cloud Scheduler job: $SCHEDULER_JOB_NAME"
    echo "This job will run on Fridays at 11:30 PM ET (30 minutes after raw data update)"
    
    gcloud scheduler jobs create http $SCHEDULER_JOB_NAME \
      --schedule="30 23 * * FRI" \
      --uri="$FUNCTION_URL" \
      --http-method=POST \
      --time-zone="America/New_York" \
      --description="Weekly parameter optimization (Fridays at 11:30 PM ET) - Runs after raw data update" \
      --headers="Content-Type=application/json" \
      --message-body='{"force_run": false}'
    
    if [ $? -eq 0 ]; then
        echo "Scheduler job created successfully!"
        echo "Job will run on Fridays at 11:30 PM ET (30 minutes after raw data update)"
        echo "This ensures raw data is complete before optimization begins"
    else
        echo "Warning: Scheduler job creation failed"
        echo "You can create it manually later if needed"
    fi
fi

echo
echo "======================================="
echo "Deployment Complete!"
echo "======================================="
echo
echo "Function Details:"
echo "- Name: $FUNCTION_NAME"
echo "- URL: $FUNCTION_URL"
echo "- Memory: $MEMORY"
echo "- Timeout: ${TIMEOUT}s"
echo
echo "Next steps:"
echo "1. Test the function manually:"
echo "   curl -X POST \"$FUNCTION_URL\" -H \"Content-Type: application/json\" -d '{\"force_run\": true, \"symbols\": [\"AAPL\"]}'"
echo
echo "2. Monitor logs:"
echo "   gcloud functions logs read $FUNCTION_NAME --region=$REGION"
echo
echo "3. The function will automatically run weekly after raw data updates"
echo "   or can be triggered manually with the curl command above"
echo
echo "4. Check Cloud Scheduler:"
echo "   gcloud scheduler jobs list"