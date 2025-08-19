#!/bin/bash

# Test Docker container functionality
# Usage: ./scripts/test-docker.sh

set -e

# Set the correct PATH for Docker Desktop on macOS
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

echo "🧪 Testing Stock Forecasting Tool Docker container..."

# Test image exists
if docker images | grep -q "stock-forecasting-app:test"; then
    echo "✅ Docker image 'stock-forecasting-app:test' found"
else
    echo "❌ Docker image 'stock-forecasting-app:test' not found"
    echo "Please build the image first: docker build -t stock-forecasting-app:test ."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found. Creating from template..."
    cp .env.example .env
    echo "🔑 Please edit .env file and add your API keys before running the container!"
fi

# Test container startup (dry run)
echo "🔍 Testing container configuration..."
if docker run --rm --env-file .env stock-forecasting-app:test python -c "
import sys
print('✅ Python version:', sys.version)

try:
    import streamlit
    print('✅ Streamlit imported successfully')
except ImportError as e:
    print('❌ Streamlit import failed:', e)
    sys.exit(1)

try:
    import pandas as pd
    print('✅ Pandas imported successfully')
except ImportError as e:
    print('❌ Pandas import failed:', e)
    sys.exit(1)

try:
    from app_modules.config import load_environment_variables
    print('✅ App modules imported successfully')
except ImportError as e:
    print('❌ App modules import failed:', e)
    sys.exit(1)

print('🎉 All imports successful!')
"; then
    echo "✅ Container test passed!"
else
    echo "❌ Container test failed!"
    exit 1
fi

# Test health check
echo "🩺 Testing container health check..."
if docker run --rm stock-forecasting-app:test curl --version > /dev/null 2>&1; then
    echo "✅ Health check tools available"
else
    echo "⚠️ Health check tools not available (this is OK for basic functionality)"
fi

echo ""
echo "🎉 Docker container tests completed successfully!"
echo ""
echo "Next steps:"
echo "1. Ensure your .env file has valid API keys"
echo "2. Run the container: docker run -p 8501:8501 --env-file .env stock-forecasting-app:test"
echo "3. Or use Docker Compose: docker-compose up -d"
echo "4. Access the app at: http://localhost:8501"
