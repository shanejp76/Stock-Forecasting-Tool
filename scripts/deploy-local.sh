#!/bin/bash

# Docker deployment script for local development
# Usage: ./scripts/deploy-local.sh

set -e

# Set the correct PATH for Docker Desktop on macOS
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

echo "🚀 Starting local deployment of Stock Forecasting Tool..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and fill in your API keys."
    exit 1
fi

# Source environment variables
source .env

# Validate required environment variables
if [ -z "$ALPHA_VANTAGE_API_KEY" ] || [ -z "$FINNHUB_API_KEY" ]; then
    echo "❌ Error: Missing required API keys in .env file!"
    echo "Please ensure ALPHA_VANTAGE_API_KEY and FINNHUB_API_KEY are set."
    exit 1
fi

echo "✅ Environment variables validated"

# Build and start the application
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting application..."
docker-compose up -d

# Wait for application to be ready
echo "⏳ Waiting for application to start..."
sleep 10

# Check if application is running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Application is running!"
    echo "🌐 Access the app at: http://localhost:8501"
    echo ""
    echo "📊 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Application failed to start. Check logs:"
    docker-compose logs
    exit 1
fi
