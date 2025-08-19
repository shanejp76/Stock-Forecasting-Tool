#!/bin/bash

# Google Cloud Run deployment script
# Usage: ./scripts/deploy-cloud.sh

set -e

PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-gcp-project-id"}
SERVICE_NAME="stock-forecasting-app"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying Stock Forecasting Tool to Google Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found!"
    echo "Please install Google Cloud SDK first."
    exit 1
fi

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "Please start Docker first."
    exit 1
fi

# Authenticate with Google Cloud
echo "🔐 Checking Google Cloud authentication..."
if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" | grep -q "@"; then
    echo "Please authenticate with Google Cloud:"
    gcloud auth login
fi

# Set the project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build the image using Cloud Build
echo "🔨 Building image with Cloud Build..."
gcloud builds submit --tag $IMAGE_NAME --file Dockerfile.prod .

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8080 \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars "ENVIRONMENT=production" \
    --set-secrets "ALPHA_VANTAGE_API_KEY=alpha-vantage-key:latest,FINNHUB_API_KEY=finnhub-key:latest"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo "✅ Deployment completed!"
echo "🌐 Your app is available at: $SERVICE_URL"
echo ""
echo "📝 Next steps:"
echo "1. Create secrets in Google Secret Manager:"
echo "   gcloud secrets create alpha-vantage-key --data-file=<(echo 'your_api_key')"
echo "   gcloud secrets create finnhub-key --data-file=<(echo 'your_api_key')"
echo "2. Update the deployment with secrets"
echo "3. Configure custom domain if needed"
