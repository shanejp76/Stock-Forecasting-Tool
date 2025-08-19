# Development Guide

This guide helps you set up and work with the Stock Forecasting Tool in a development environment.

## Quick Setup

### Prerequisites
- Python 3.11
- Docker and Docker Compose (optional)
- Git

### Automated Setup
```bash
# Clone the repository (if not already done)
git clone https://github.com/yourusername/Stock-Forecasting-Tool.git
cd Stock-Forecasting-Tool

# Run the setup script
./scripts/setup-dev.sh
```

### Manual Setup
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Environment Management
```bash
# Activate virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

### 2. Code Quality Tools
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy app_modules/

# Run all quality checks
pre-commit run --all-files
```

### 3. Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app_modules --cov-report=html

# Run specific test file
pytest tests/test_data_handler.py

# Run with verbose output
pytest -v
```

### 4. Running the Application
```bash
# Local development
streamlit run main.py

# With Docker
./scripts/deploy-local.sh

# Production-like build
docker build -f Dockerfile.prod -t stock-app:latest .
docker run -p 8080:8080 --env-file .env stock-app:latest
```

## Project Structure

```
Stock-Forecasting-Tool/
├── main.py                    # Main Streamlit application
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── .env.example              # Environment variables template
├── Dockerfile                # Development Docker image
├── Dockerfile.prod           # Production Docker image
├── docker-compose.yml        # Docker Compose configuration
├── .dockerignore            # Docker ignore patterns
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── .github/
│   └── workflows/
│       └── ci-cd.yml        # GitHub Actions CI/CD
├── app_modules/             # Application modules
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── data_handler.py     # Data processing
│   ├── data_pipeline.py    # Data pipeline functions
│   ├── model_orchestrator.py  # Model training orchestration
│   ├── plotter.py          # Visualization functions
│   └── ...                 # Other modules
├── tests/                   # Test suite
│   ├── conftest.py         # Test configuration
│   ├── test_data_handler.py # Data handler tests
│   └── test_integration.py # Integration tests
├── scripts/                 # Deployment and utility scripts
│   ├── setup-dev.sh        # Development setup
│   ├── deploy-local.sh     # Local Docker deployment
│   └── deploy-cloud.sh     # Cloud deployment
├── docs/                    # Documentation
│   ├── MODERNIZATION_ROADMAP.md
│   ├── DOCKER_DEPLOYMENT.md
│   └── DEVELOPMENT_GUIDE.md
└── Archive/                 # Historical development files
```

## Key Development Concepts

### 1. Modular Architecture
- **Separation of Concerns**: Each module handles specific functionality
- **Data Pipeline**: Clear flow from data ingestion to visualization
- **UI Components**: Reusable Streamlit interface components
- **Configuration**: Centralized environment and parameter management

### 2. Data Flow
```
API Data → Data Handler → Technical Indicators → Model Training → Forecasting → Visualization
```

### 3. Testing Strategy
- **Unit Tests**: Individual function testing
- **Integration Tests**: Module interaction testing
- **End-to-End Tests**: Full application workflow testing
- **Performance Tests**: Load and stress testing

## Common Development Tasks

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes in appropriate modules
3. Add tests for new functionality
4. Update documentation
5. Run quality checks: `pre-commit run --all-files`
6. Create pull request

### Adding New Dependencies
1. Add to `requirements.txt` (production) or `requirements-dev.txt` (development)
2. Update Docker images if needed
3. Update CI/CD pipeline if necessary
4. Document any breaking changes

### Debugging Common Issues

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify module structure
ls -la app_modules/

# Test specific imports
python -c "from app_modules.config import load_environment_variables"
```

#### API Connection Issues
```bash
# Check environment variables
env | grep API

# Test API connectivity
python -c "
from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='your_key_here')
print('API connection successful')
"
```

#### Docker Issues
```bash
# Check Docker status
docker info

# View container logs
docker-compose logs streamlit-app

# Debug container
docker run -it --entrypoint /bin/bash stock-app:latest
```

## Performance Optimization

### 1. Streamlit Optimization
- Use `@st.cache_data` for expensive computations
- Implement session state for user data
- Optimize data loading with pagination
- Use `st.empty()` for dynamic content updates

### 2. Data Processing
- Vectorize operations with pandas/numpy
- Use efficient data types (category, int32 vs int64)
- Implement data sampling for large datasets
- Cache preprocessed data

### 3. Model Training
- Use cross-validation efficiently
- Implement early stopping
- Parallelize hyperparameter tuning
- Cache trained models

## Monitoring and Debugging

### Application Metrics
```python
# Add to your code for monitoring
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def expensive_function():
    start_time = time.time()
    # Your code here
    logger.info(f"Function took {time.time() - start_time:.2f} seconds")
```

### Health Checks
```bash
# Application health
curl http://localhost:8501/_stcore/health

# Container health
docker inspect stock-app --format='{{.State.Health.Status}}'
```

## Deployment Pipeline

### Local Development
1. Make changes
2. Run tests: `pytest`
3. Run quality checks: `pre-commit run --all-files`
4. Test locally: `streamlit run main.py`

### Staging Deployment
1. Push to feature branch
2. Create pull request
3. CI/CD runs automatically
4. Review staging deployment
5. Merge to main

### Production Deployment
1. Merge to main branch
2. Automated CI/CD pipeline
3. Deploy to production
4. Monitor application health

## Best Practices

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Keep functions small and focused
- Use type hints where appropriate

### Git Workflow
- Use descriptive commit messages
- Create feature branches for new work
- Squash commits before merging
- Use conventional commit format

### Security
- Never commit API keys or secrets
- Use environment variables for configuration
- Regularly update dependencies
- Follow principle of least privilege

### Documentation
- Keep README up to date
- Document complex algorithms
- Add inline comments for clarity
- Update API documentation

## Troubleshooting

### Common Errors

1. **ModuleNotFoundError**
   - Check virtual environment activation
   - Verify PYTHONPATH
   - Reinstall dependencies

2. **API Rate Limits**
   - Implement caching
   - Add retry logic with backoff
   - Use multiple API keys if available

3. **Memory Issues**
   - Implement data chunking
   - Use generators for large datasets
   - Clear unused variables

4. **Performance Issues**
   - Profile code with cProfile
   - Use Streamlit's profiler
   - Optimize database queries

### Getting Help
1. Check existing issues on GitHub
2. Review documentation
3. Ask in community forums
4. Create detailed bug reports

---

Happy coding! 🚀
