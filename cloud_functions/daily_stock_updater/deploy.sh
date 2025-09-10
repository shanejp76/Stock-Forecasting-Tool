#!/bin/bash

# Deploy Daily Stock Updater Cloud Function
# Usage: ./deploy.sh [your-alpha-vantage-api-key] [your-gcp-project-id]

set -e

# Configuration
FUNCTION_NAME="daily-stock-update"
REGION="us-central1"  # Change as needed
RUNTIME="python311"
MEMORY="512MB"
TIMEOUT="540s"

# Get parameters
ALPHA_VANTAGE_API_KEY=${1:-$ALPHA_VANTAGE_API_KEY}
BIGQUERY_PROJECT_ID=${2:-$GOOGLE_CLOUD_PROJECT}

if [ -z "$ALPHA_VANTAGE_API_KEY" ]; then
    echo "Error: ALPHA_VANTAGE_API_KEY is required"
    echo "Usage: $0 [alpha-vantage-key] [gcp-project-id]"
    echo "Or set environment variables ALPHA_VANTAGE_API_KEY and GOOGLE_CLOUD_PROJECT"
    exit 1
fi

if [ -z "$BIGQUERY_PROJECT_ID" ]; then
    echo "Error: BIGQUERY_PROJECT_ID is required"
    echo "Usage: $0 [alpha-vantage-key] [gcp-project-id]"
    echo "Or set environment variables ALPHA_VANTAGE_API_KEY and GOOGLE_CLOUD_PROJECT"
    exit 1
fi

echo "Deploying Cloud Function: $FUNCTION_NAME"
echo "Project: $BIGQUERY_PROJECT_ID"
echo "Region: $REGION"

# Deploy the function
gcloud functions deploy $FUNCTION_NAME \
  --runtime $RUNTIME \
  --trigger-http \
  --allow-unauthenticated \
  --region $REGION \
  --memory $MEMORY \
  --timeout $TIMEOUT \
  --set-env-vars "ALPHA_VANTAGE_API_KEY=$ALPHA_VANTAGE_API_KEY,BIGQUERY_PROJECT_ID=$BIGQUERY_PROJECT_ID,MAX_TRADING_DAYS=500" \
  --source . \
  --entry-point daily_stock_update

echo "Function deployed successfully!"

# Get the function URL
FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME --region=$REGION --format="value(httpsTrigger.url)")
echo "Function URL: $FUNCTION_URL"

# Optionally create a Cloud Scheduler job
read -p "Create a daily Cloud Scheduler job? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SCHEDULER_JOB_NAME="daily-stock-update-job"
    
    echo "Creating Cloud Scheduler job: $SCHEDULER_JOB_NAME"
    
    gcloud scheduler jobs create http $SCHEDULER_JOB_NAME \
      --schedule="0 22 * * MON-FRI" \
      --uri="$FUNCTION_URL" \
      --http-method=POST \
      --time-zone="America/New_York" \
      --description="Daily stock data update (weekdays at 10 PM ET) - Auto-discovers symbols from BigQuery" \
      --headers="Content-Type=application/json" \
      --message-body='{}'
    
    echo "Scheduler job created successfully!"
    echo "Job will run weekdays at 10 PM ET (after market close)"
fi

echo "Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Test the function: curl -X POST $FUNCTION_URL"
echo "2. Monitor logs: gcloud functions logs read $FUNCTION_NAME --region=$REGION"
echo "3. Update your Streamlit app to use the fresh BigQuery data"
