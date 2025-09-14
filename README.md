# Stock Forecasting Tool

A modern, containerized analytics engineering platform for stock forecasting and swing trading analysis. This application combines machine learning, technical analysis, and business intelligence to provide comprehensive insights for trading decisions.

<!-- CI/CD Status Check - Updated: 2025-09-13 - Testing Cloud Run deployment permissions -->

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your Alpha Vantage and Finnhub API keys

# Deploy with Docker
./scripts/deploy-local.sh

# Access at http://localhost:8501
```

### Option 2: Local Development
```bash
# Setup development environment
./scripts/setup-dev.sh

# Run the application
streamlit run main.py
```

## 📋 Prerequisites

- **API Keys**: Alpha Vantage and Finnhub API keys
- **Docker**: For containerized deployment (recommended)
- **Python 3.11**: For local development
- **Git**: For version control
- **Google Cloud**: For BigQuery data warehouse features

## 🔐 Setup for New Development Environment

### Required Files (Not in Git Repository)
Due to security, some files are not stored in git and must be set up manually:

- `credentials.json` - Google Cloud service account key for BigQuery access
- `.env` - Environment variables with your API keys

### Quick Setup Process
1. **Clone the repository**
   ```bash
   git clone https://github.com/shanejp76/Stock-Forecasting-Tool.git
   cd Stock-Forecasting-Tool
   ```

2. **Run the automated setup script**
   ```bash
   ./scripts/setup-dev.sh
   ```

3. **Set up Google Cloud credentials** (choose one option):
   
   **Option A: Copy from existing environment**
   ```bash
   # Copy credentials.json from another development machine
   # (transfer securely - never email or store in unsecured locations)
   ```
   
   **Option B: Generate new credentials**
   ```bash
   # Authenticate with Google Cloud
   gcloud auth login
   gcloud config set project stock-forecasting-tool-2025
   
   # Create new service account key
   gcloud iam service-accounts keys create credentials.json \
     --iam-account=stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com
   ```

4. **Configure API keys**
   ```bash
   # Edit .env file with your actual API keys
   cp .env.example .env
   # Add your Alpha Vantage and Finnhub API keys
   ```

5. **Test the setup**
   ```bash
   # Test application
   streamlit run main.py
   
   # Test BigQuery connection
   python3 scripts/initial_bulk_load.py --symbols AAPL --yes
   ```

### Security Notes
- Never commit `credentials.json` to git (already in .gitignore)
- Each development environment should ideally have its own service account key
- Rotate credentials regularly
- Store credentials securely when transferring between machines

## ✨ Features

### Core Functionality
- **Advanced Forecasting**: Custom-tuned Prophet model with 15% median SMAPE
- **Technical Analysis**: SMAs, Bollinger Bands, RSI, MACD, Golden/Death Cross
- **Market Correlation**: Risk assessment against market indices
- **Volatility Analysis**: Dynamic training periods and outlier handling
- **Business KPIs**: Actionable insights for trading decisions

### Modern Architecture
- **🐳 Containerized**: Docker support for consistent deployment
- **☁️ Cloud Ready**: Google Cloud Run deployment scripts
- **🔄 CI/CD Pipeline**: Automated testing and deployment
- **📊 Analytics Engineering**: BigQuery and dbt integration ready
- **🧪 Tested**: Comprehensive test suite with quality checks

## 🏗️ Project Structure

```
Stock-Forecasting-Tool/
├── 📱 main.py                     # Streamlit application
├── 🐳 Dockerfile                  # Development container
├── 🐳 Dockerfile.prod            # Production container
├── 🐙 docker-compose.yml         # Local development stack
├── 📦 requirements.txt           # Python dependencies
├── ⚙️ .env.example              # Environment template
├── 🔧 app_modules/              # Application modules
│   ├── config.py               # Configuration management
│   ├── data_handler.py         # Data processing
│   ├── model_orchestrator.py   # ML pipeline
│   └── ...                     # Other modules
├── 🧪 tests/                    # Test suite
├── 📜 scripts/                  # Deployment scripts
├── 📚 docs/                     # Documentation
└── 🤖 .github/workflows/       # CI/CD pipelines
```

## 🛠️ Development

### Setup Development Environment
```bash
# Automated setup
./scripts/setup-dev.sh

# Or manual setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt requirements-dev.txt
```

### Code Quality Tools
```bash
# Format and lint
black .
isort .
flake8 .

# Run tests
pytest

# Pre-commit hooks
pre-commit install
```

## 🚀 Deployment Options

### Local Development
```bash
./scripts/deploy-local.sh
```

### Cloud Deployment (Google Cloud Run)
```bash
./scripts/deploy-cloud.sh
```

### Manual Docker
```bash
# Development
docker-compose up -d

# Production
docker build -f Dockerfile.prod -t stock-app .
docker run -p 8080:8080 --env-file .env stock-app
```

## 📈 Analytics Engineering Roadmap

This project is being transformed into a modern analytics engineering platform:

### Phase 1: ✅ Containerization (Complete)
- Docker containerization
- CI/CD pipeline
- Cloud deployment ready

### Phase 2: 🔄 Data Warehouse Integration (In Progress)
- Google BigQuery integration
- Centralized data storage
- Historical data persistence

### Phase 3: 📊 dbt Transformation Layer (Planned)
- SQL-based transformations
- Data quality testing
- Version-controlled data models

### Phase 4: 🔀 Orchestration (Planned)
- Mage AI workflow orchestration
- Automated data pipelines
- Monitoring and alerting

See [MODERNIZATION_ROADMAP.md](MODERNIZATION_ROADMAP.md) for detailed implementation plan.

## 📊 Model Performance

- **Accuracy**: 15% median SMAPE across 150 diverse stock tickers
- **Validation**: Wilcoxon signed-rank test (p < 0.05)
- **Effect Size**: Large effect size (Cliff's Delta = 0.69) vs standard Prophet
- **Robustness**: Volatility-adjusted winsorization for outlier handling

## 🔧 Configuration

### Environment Variables
```bash
# Required API Keys
ALPHA_VANTAGE_API_KEY=your_api_key
FINNHUB_API_KEY=your_api_key

# Optional Configuration
STREAMLIT_SERVER_PORT=8501
ENVIRONMENT=development
```

### Application Settings
- Adjustable forecast periods
- Customizable technical indicators
- Configurable model parameters
- Dynamic training windows

## 📚 Documentation

- [🐳 Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md)
- [🛠️ Development Guide](docs/DEVELOPMENT_GUIDE.md)
- [🗺️ Modernization Roadmap](docs/MODERNIZATION_ROADMAP.md)
- [📋 Containerization Summary](docs/CONTAINERIZATION_SUMMARY.md)
- [🎉 Completion Summary](docs/COMPLETION_SUMMARY.md)
- [📖 Original Documentation](docs/Forecasting%20Tool%20Documentation.pdf)

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app_modules --cov-report=html

# Integration tests only
pytest tests/test_integration.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `pre-commit run --all-files`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 Check the [documentation](docs/)
- 🐛 Report issues on [GitHub Issues](https://github.com/shanejp76/Stock-Forecasting-Tool/issues)
- 💬 Join discussions in [GitHub Discussions](https://github.com/shanejp76/Stock-Forecasting-Tool/discussions)

## 🙏 Acknowledgments

- Prophet forecasting library by Facebook
- Streamlit for the interactive web framework
- Alpha Vantage and Finnhub for market data APIs
- The open-source community for tools and inspiration

---

**Made with ❤️ for the trading and data science community**
# Deployment test Mon Aug 25 12:45:27 PDT 2025
# APIs enabled - ready for deployment Mon Aug 25 12:53:21 PDT 2025
# APIs enabled - deployment ready Mon Aug 25 12:57:03 PDT 2025
