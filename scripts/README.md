# Scripts Directory

This directory contains utility scripts for development, deployment, and testing of the Stock Forecasting Tool.

## 📜 Available Scripts

### 🛠️ Development
- **`setup-dev.sh`** - Automated development environment setup
  ```bash
  ./scripts/setup-dev.sh
  ```
  - Creates virtual environment
  - Installs all dependencies
  - Sets up pre-commit hooks
  - Creates necessary directories

### 🐳 Docker & Deployment
- **`deploy-local.sh`** - Local Docker deployment
  ```bash
  ./scripts/deploy-local.sh
  ```
  - Validates environment variables
  - Builds and runs Docker containers locally
  - Provides access URL and management commands

- **`deploy-cloud.sh`** - Google Cloud Run deployment
  ```bash
  ./scripts/deploy-cloud.sh
  ```
  - Builds container image with Cloud Build
  - Deploys to Google Cloud Run
  - Configures secrets and environment variables

- **`docker-wrapper.sh`** - Docker helper for macOS
  ```bash
  ./scripts/docker-wrapper.sh [docker-commands]
  ```
  - Sets correct PATH for Docker Desktop on macOS
  - Validates Docker is running
  - Executes Docker commands with proper environment

### 🧪 Testing
- **`test-docker.sh`** - Docker container testing
  ```bash
  ./scripts/test-docker.sh
  ```
  - Validates Docker image exists
  - Tests container functionality
  - Verifies imports and dependencies

## 🚀 Quick Start Workflows

### First Time Setup
```bash
# 1. Setup development environment
./scripts/setup-dev.sh

# 2. Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Test with Docker
./scripts/deploy-local.sh
```

### Development Workflow
```bash
# Setup (once)
./scripts/setup-dev.sh

# Daily development
source venv/bin/activate
streamlit run main.py

# Testing
./scripts/test-docker.sh
```

### Deployment Workflow
```bash
# Local testing
./scripts/deploy-local.sh

# Cloud deployment
./scripts/deploy-cloud.sh
```

## 🔧 Script Requirements

### System Requirements
- **macOS/Linux**: All scripts designed for Unix-like systems
- **Docker**: Required for containerization scripts
- **Python 3.11**: Required for development scripts
- **Google Cloud SDK**: Required for cloud deployment

### Environment Variables
Scripts expect certain environment variables to be set in `.env`:
```bash
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
GOOGLE_CLOUD_PROJECT=your_project_id  # For cloud deployment
```

## 🛠️ Script Permissions

All scripts should be executable. If you encounter permission issues:
```bash
chmod +x scripts/*.sh
```

## 📋 Troubleshooting

### Common Issues

1. **"Permission denied"**
   ```bash
   chmod +x scripts/script-name.sh
   ```

2. **"Docker not found" (macOS)**
   - Use `docker-wrapper.sh` or ensure Docker Desktop is in PATH
   ```bash
   export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
   ```

3. **"Python 3.11 not found"**
   - Install Python 3.11 or modify scripts to use available Python version

4. **"API key not found"**
   - Ensure `.env` file exists and contains valid API keys
   - Copy from `.env.example` if needed

### Getting Help
- Check individual script comments for detailed usage
- Refer to documentation in `../docs/` directory
- Create GitHub issue for persistent problems

---

**Note**: All scripts include built-in help and error checking. Run any script to see its current status and requirements.
