#!/bin/bash
# Development Environment Setup Script for Stock Forecasting Tool
# This script sets up a new development environment with all required dependencies

set -e  # Exit on any error

echo "=========================================="
echo "Stock Forecasting Tool - Development Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the project root
if [ ! -f "main.py" ] || [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_status "Starting development environment setup..."

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
required_version="3.11"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    print_success "Python $python_version detected (>= $required_version required)"
else
    print_error "Python 3.11+ required. Current version: $python_version"
    print_status "Please install Python 3.11+ and try again"
    exit 1
fi

# Check for Google Cloud SDK
print_status "Checking Google Cloud SDK..."
if command -v gcloud >/dev/null 2>&1; then
    gcloud_version=$(gcloud version 2>/dev/null | head -n1 | cut -d' ' -f4)
    print_success "Google Cloud SDK detected: $gcloud_version"
else
    print_warning "Google Cloud SDK not found"
    print_status "Installing Google Cloud SDK..."
    
    # Detect OS and install accordingly
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install --cask google-cloud-sdk
        else
            print_status "Please install Homebrew first, then run: brew install --cask google-cloud-sdk"
            print_status "Or install manually: https://cloud.google.com/sdk/docs/install-sdk"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl https://sdk.cloud.google.com | bash
        exec -l $SHELL
    else
        print_error "Unsupported OS. Please install Google Cloud SDK manually:"
        print_status "https://cloud.google.com/sdk/docs/install"
    fi
fi

# Check Docker
print_status "Checking Docker..."
if command -v docker >/dev/null 2>&1; then
    docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    print_success "Docker detected: $docker_version"
    
    # Test Docker daemon
    if docker ps >/dev/null 2>&1; then
        print_success "Docker daemon is running"
    else
        print_warning "Docker daemon is not running"
        print_status "Please start Docker Desktop or Docker daemon"
    fi
else
    print_warning "Docker not found"
    print_status "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    print_success "Python dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Install development dependencies
if [ -f "requirements-dev.txt" ]; then
    print_status "Installing development dependencies..."
    pip3 install -r requirements-dev.txt
    print_success "Development dependencies installed"
fi

# Check for credentials file
print_status "Checking for Google Cloud credentials..."
if [ -f "credentials.json" ]; then
    print_success "credentials.json found"
    
    # Test credentials
    print_status "Testing BigQuery connection..."
    if python3 -c "
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
from app_modules.bigquery_client import BigQueryClient
try:
    client = BigQueryClient()
    print('BigQuery connection successful!')
except Exception as e:
    print(f'BigQuery connection failed: {e}')
    exit(1)
" 2>/dev/null; then
        print_success "BigQuery connection test passed"
    else
        print_error "BigQuery connection test failed"
        print_status "Please check your credentials.json file"
    fi
else
    print_warning "credentials.json not found"
    print_status "You need to either:"
    print_status "1. Copy credentials.json from another development environment"
    print_status "2. Generate new credentials with:"
    echo ""
    echo "   gcloud auth login"
    echo "   gcloud config set project stock-forecasting-tool-2025"
    echo "   gcloud iam service-accounts keys create credentials.json \\"
    echo "     --iam-account=stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com"
    echo ""
fi

# Check environment file
print_status "Checking environment configuration..."
if [ -f ".env" ]; then
    print_success ".env file found"
    
    # Check for required variables
    if grep -q "ALPHA_VANTAGE_API_KEY" .env && grep -q "GOOGLE_CLOUD_PROJECT" .env; then
        print_success "Required environment variables present"
        
        # Check if they still have placeholder values
        if grep -q "your_alpha_vantage_api_key_here" .env; then
            print_warning "Alpha Vantage API key appears to be placeholder"
            print_status "Please update .env with your actual Alpha Vantage API key"
        fi
        
        if grep -q "your_gcp_project_id" .env; then
            print_warning "GCP project ID appears to be placeholder"
            print_status "Please update .env with: GOOGLE_CLOUD_PROJECT=stock-forecasting-tool-2025"
        fi
    else
        print_warning "Some required environment variables may be missing"
        print_status "Please check .env file contains:"
        print_status "- ALPHA_VANTAGE_API_KEY"
        print_status "- FINNHUB_API_KEY"
        print_status "- GOOGLE_CLOUD_PROJECT=stock-forecasting-tool-2025"
    fi
else
    print_warning ".env file not found"
    print_status "Creating .env from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        # Update with correct project ID
        sed -i '' 's/your_gcp_project_id/stock-forecasting-tool-2025/g' .env
        print_success ".env created from template with correct GCP project"
        print_status "Please edit .env file with your API keys"
    else
        print_error ".env.example not found"
    fi
fi

# Create data directory
print_status "Setting up data directory..."
mkdir -p data
mkdir -p logs
mkdir -p reports
if [ ! -f "data/.gitkeep" ]; then
    touch data/.gitkeep
    print_success "Data directory created"
fi

# Setup pre-commit hooks if available
if command -v pre-commit >/dev/null 2>&1; then
    print_status "Setting up pre-commit hooks..."
    if [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning ".pre-commit-config.yaml not found"
    fi
else
    print_status "Pre-commit not found, installing..."
    pip3 install pre-commit
    if [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    fi
fi

# Check for pytest configuration
print_status "Checking test configuration..."
if [ -f "pytest.ini" ]; then
    print_success "pytest.ini found"
else
    print_warning "pytest.ini not found"
fi

# Check Docker configuration
print_status "Checking Docker configuration..."
if [ -f "docker-compose.yml" ]; then
    print_success "docker-compose.yml found"
else
    print_warning "docker-compose.yml not found"
fi

if [ -f "Dockerfile" ] && [ -f "Dockerfile.prod" ]; then
    print_success "Docker files found"
else
    print_warning "Some Docker files missing"
fi

# Test application startup
print_status "Testing application components..."

# Check main application modules
missing_modules=()
required_modules=(
    "app_modules/config.py"
    "app_modules/data_handler.py" 
    "app_modules/bigquery_client.py"
    "app_modules/model_orchestrator.py"
    "app_modules/rate_limiter.py"
)

for module in "${required_modules[@]}"; do
    if [ ! -f "$module" ]; then
        missing_modules+=("$module")
    fi
done

if [ ${#missing_modules[@]} -eq 0 ]; then
    print_success "All required modules found"
else
    print_warning "Missing modules: ${missing_modules[*]}"
fi

# Check important scripts
required_scripts=(
    "scripts/initial_bulk_load.py"
    "scripts/deploy-local.sh"
    "scripts/deploy-cloud.sh"
)

missing_scripts=()
for script in "${required_scripts[@]}"; do
    if [ ! -f "$script" ]; then
        missing_scripts+=("$script")
    fi
done

if [ ${#missing_scripts[@]} -eq 0 ]; then
    print_success "All required scripts found"
else
    print_warning "Missing scripts: ${missing_scripts[*]}"
fi

# Test Python imports
if python3 -c "
import sys
sys.path.append('.')
try:
    from app_modules.config import load_environment_variables
    from app_modules.bigquery_client import BigQueryClient
    from app_modules.data_handler import DataHandler
    print('Core module imports successful!')
except Exception as e:
    print(f'Module import failed: {e}')
    exit(1)
" 2>/dev/null; then
    print_success "Application module test passed"
else
    print_warning "Application module test failed - this may be normal if credentials are missing"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
print_success "Development environment setup completed!"
echo ""
print_status "Next steps:"
echo ""
if [ ! -f "credentials.json" ]; then
    echo "1. REQUIRED: Set up Google Cloud credentials"
    echo "   gcloud auth login"
    echo "   gcloud config set project stock-forecasting-tool-2025" 
    echo "   gcloud iam service-accounts keys create credentials.json \\"
    echo "     --iam-account=stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com"
    echo ""
fi

if [ ! -f ".env" ] || grep -q "your_alpha_vantage_api_key_here" .env 2>/dev/null; then
    echo "2. REQUIRED: Update .env file with your actual API keys"
    echo "   Edit .env and replace placeholder values with real API keys"
    echo ""
fi

echo "3. Test the application:"
echo "   streamlit run main.py"
echo "   # Access at http://localhost:8501"
echo ""

echo "4. For Docker development:"
echo "   ./scripts/deploy-local.sh"
echo "   # or manually:"
echo "   docker-compose up -d"
echo ""

echo "5. For BigQuery data loading:"
echo "   python3 scripts/initial_bulk_load.py --symbols AAPL --yes"
echo ""

echo "6. Development workflow:"
echo "   # Format code"
echo "   black ."
echo "   isort ."
echo ""
echo "   # Run tests"
echo "   pytest"
echo ""
echo "   # Lint code"
echo "   flake8 ."
echo ""

echo "7. Deploy to cloud:"
echo "   ./scripts/deploy-cloud.sh"
echo ""

print_status "Useful commands:"
echo "- View logs: docker-compose logs -f"
echo "- Stop containers: docker-compose down"
echo "- Rebuild: docker-compose build --no-cache"
echo "- Shell into container: docker-compose exec app bash"
echo ""

if [ ${#missing_modules[@]} -gt 0 ] || [ ${#missing_scripts[@]} -gt 0 ]; then
    print_warning "Some files are missing. Please check your git clone or repository integrity."
fi
echo ""
print_success "Happy coding!"
