#!/bin/bash

# Development environment setup script
# Usage: ./scripts/setup-dev.sh

set -e

echo "🚀 Setting up Stock Forecasting Tool development environment..."

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 not found. Please install Python 3.11 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install production dependencies
echo "📚 Installing production dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -r requirements-dev.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
    echo "🔑 Please edit .env file and add your API keys!"
fi

# Setup pre-commit hooks
echo "🔒 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p logs
mkdir -p reports

# Check Docker installation
if command -v docker &> /dev/null; then
    echo "🐳 Docker is available"
    if command -v docker-compose &> /dev/null; then
        echo "🐙 Docker Compose is available"
    else
        echo "⚠️ Docker Compose not found. Please install it for containerized development."
    fi
else
    echo "⚠️ Docker not found. Please install Docker for containerization features."
fi

# Test basic imports
echo "🧪 Testing basic imports..."
python -c "
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    print('✅ Core dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Run basic tests
echo "🧪 Running basic tests..."
if python -m pytest tests/test_integration.py -v; then
    echo "✅ Basic tests passed"
else
    echo "⚠️ Some tests failed, but environment setup is complete"
fi

echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your API keys"
echo "3. Run the app: streamlit run main.py"
echo "4. Or use Docker: ./scripts/deploy-local.sh"
echo ""
echo "Development commands:"
echo "- Run tests: pytest"
echo "- Format code: black ."
echo "- Sort imports: isort ."
echo "- Lint code: flake8 ."
echo "- Type check: mypy ."
