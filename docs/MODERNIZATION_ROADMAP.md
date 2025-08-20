# Stock Forecasting Tool - Modernization Roadmap

## Overview
This document outlines the transformation of the existing Streamlit-based stock forecasting application into a modern, production-ready analytics engineering platform. The modernization will introduce cloud data warehousing, proper data transformation pipelines, and full automation.

## Current Architecture
- **Frontend**: Streamlit web application
- **Data Source**: Direct API calls to stock data providers
- **Processing**: Real-time data processing within the app
- **Storage**: Local/temporary data handling
- **Deployment**: Local execution

## Target Architecture
- **Data Warehouse**: Google BigQuery for centralized data storage
- **Data Transformation**: dbt (data build tool) for SQL-based transformations
- **Orchestration**: Mage AI for workflow scheduling
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Frontend**: Enhanced Streamlit app querying transformed data
- **Deployment**: Containerized with Docker

## Phase 1: Cloud Data Warehouse Integration

### Objectives
- Replace direct API calls with data warehouse queries
- Implement proper data ingestion pipeline
- Separate data concerns from application logic

### Implementation Steps
1. **Setup Google Cloud BigQuery**
   - Create BigQuery dataset for stock data
   - Configure service account and authentication
   - Design raw data schema

2. **Data Ingestion Pipeline**
   - Modify existing Python scripts to load data into BigQuery
   - Implement incremental data loading
   - Add data validation and error handling

3. **Raw Data Tables**
   ```sql
   -- Example schema
   CREATE TABLE raw_stock_data (
     symbol STRING,
     timestamp TIMESTAMP,
     open FLOAT64,
     high FLOAT64,
     low FLOAT64,
     close FLOAT64,
     volume INT64,
     ingested_at TIMESTAMP
   );
   ```

### Benefits
- Centralized data storage
- Historical data persistence
- Scalable data architecture
- Separation of concerns

## Phase 2: dbt Data Transformation Layer

### Objectives
- Move feature engineering from Python to SQL
- Implement version-controlled transformation logic
- Create clean, documented data models

### Implementation Steps
1. **dbt Project Setup**
   - Initialize dbt project structure
   - Configure BigQuery connection
   - Set up development and production environments

2. **Data Models**
   - **Staging Models**: Clean and standardize raw data
   - **Intermediate Models**: Feature engineering and calculations
   - **Mart Models**: Final datasets for forecasting and analysis

3. **Key Transformations**
   ```sql
   -- Example staging model
   {{ config(materialized='view') }}
   
   SELECT
     symbol,
     timestamp,
     close,
     LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_close,
     (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp)) / LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp) as daily_return
   FROM {{ ref('raw_stock_data') }}
   ```

4. **Data Testing**
   - Implement dbt tests for data quality
   - Custom tests for business logic validation
   - Freshness and uniqueness checks

### Benefits
- SQL-based transformations (faster, more maintainable)
- Version control for data logic
- Automated testing and documentation
- Lineage tracking

## Phase 3: Application Modernization

### Objectives
- Refactor Streamlit app to use transformed data
- Improve performance and user experience
- Add advanced analytics features

### Implementation Steps
1. **Database Connection Layer**
   - Implement BigQuery client in Streamlit app
   - Create data access layer for transformed tables
   - Add caching for improved performance

2. **Enhanced Features**
   - Real-time dashboard with historical context
   - Advanced forecasting models using warehouse data
   - Performance benchmarking and model comparison

3. **Configuration Management**
   - Environment-specific configurations
   - Secrets management for API keys and credentials

## Phase 4: Automation & Orchestration

### Objectives
- Automate the entire data pipeline
- Implement CI/CD for code and data
- Schedule regular data updates and model retraining

### Implementation Steps
1. **GitHub Actions CI/CD**
   ```yaml
   # .github/workflows/dbt-ci.yml
   name: dbt CI/CD
   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]
   
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Setup dbt
           run: pip install dbt-bigquery
         - name: Run dbt tests
           run: dbt test
   ```

2. **Mage AI Orchestration**
   - Data ingestion pipelines
   - dbt model execution
   - Model training and evaluation
   - Alert and monitoring setup

3. **Monitoring & Alerting**
   - Data quality monitoring
   - Pipeline failure alerts
   - Performance metrics tracking

## Phase 5: Containerization & Deployment

### Objectives
- Containerize all components for consistent deployment
- Enable cloud deployment
- Implement proper environment management

### Implementation Steps
1. **Docker Configuration**
   - Multi-stage builds for optimization
   - Separate containers for different components
   - Docker Compose for local development

2. **Cloud Deployment**
   - Google Cloud Run for Streamlit app
   - Cloud Functions for data ingestion
   - Cloud Scheduler for automation

## Current Status: Ready for GCP Deployment

### CI/CD Pipeline Status ✅
The containerization and CI/CD pipeline infrastructure is complete and fully functional:
- ✅ **Docker containerization**: Multi-stage builds with production optimization
- ✅ **GitHub Actions**: Automated testing, security scanning, and container builds
- ✅ **Container Registry**: Images automatically built and pushed to GitHub Container Registry
- ✅ **Local deployment**: Verified working at `http://localhost:8501`

### Next Steps: Google Cloud Platform Setup

To enable cloud deployment, the following GCP configuration steps are required:

#### 1. Google Cloud Project Setup
```bash
# Create new project (or use existing)
gcloud projects create stock-forecasting-tool-prod
gcloud config set project stock-forecasting-tool-prod

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

#### 2. Service Account Configuration
```bash
# Create service account for GitHub Actions
gcloud iam service-accounts create github-actions \
    --description="Service account for GitHub Actions CI/CD" \
    --display-name="GitHub Actions"

# Grant necessary permissions
gcloud projects add-iam-policy-binding stock-forecasting-tool-prod \
    --member="serviceAccount:github-actions@stock-forecasting-tool-prod.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding stock-forecasting-tool-prod \
    --member="serviceAccount:github-actions@stock-forecasting-tool-prod.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Create and download service account key
gcloud iam service-accounts keys create github-actions-key.json \
    --iam-account=github-actions@stock-forecasting-tool-prod.iam.gserviceaccount.com
```

#### 3. GitHub Repository Secrets
Add the following secrets to your GitHub repository (`Settings > Secrets and variables > Actions`):

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `GCP_SERVICE_ACCOUNT_KEY` | Contents of `github-actions-key.json` | Base64-encoded service account key |
| `GCP_PROJECT_ID` | `stock-forecasting-tool-prod` | Google Cloud project ID |
| `GCP_REGION` | `us-central1` | Deployment region |

#### 4. Enable Deployment Jobs
Once the secrets are configured, uncomment the deployment jobs in `.github/workflows/ci-cd.yml`:

```yaml
# Uncomment these jobs in ci-cd.yml after GCP setup
deploy-staging:
  needs: build-and-push
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  environment: staging
  
deploy-production:
  needs: deploy-staging
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  environment: production
```

#### 5. Cloud Run Deployment Configuration
The deployment will create:
- **Staging environment**: `stock-forecasting-staging` service
- **Production environment**: `stock-forecasting-prod` service
- **Custom domains**: Can be configured post-deployment
- **Environment variables**: API keys and configuration injected securely

#### 6. Verification Steps
After GCP setup:
1. Push code to trigger CI/CD pipeline
2. Verify deployment jobs complete successfully
3. Access staging URL provided in deployment logs
4. Test application functionality in cloud environment
5. Promote to production if staging tests pass

### Estimated Setup Time
- **GCP Project Setup**: 15 minutes
- **Service Account Configuration**: 10 minutes  
- **GitHub Secrets Configuration**: 5 minutes
- **First Deployment**: 10 minutes
- **Testing & Verification**: 15 minutes

**Total: ~1 hour for complete cloud deployment setup**

## Technology Stack

### Core Technologies
- **Data Warehouse**: Google BigQuery
- **Transformation**: dbt (data build tool)
- **Orchestration**: Mage AI
- **Frontend**: Streamlit
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

### Supporting Tools
- **Authentication**: Google Cloud IAM
- **Monitoring**: Google Cloud Monitoring
- **Version Control**: Git/GitHub
- **Environment Management**: Python virtual environments

## Benefits of Modernization

### Technical Benefits
- **Scalability**: Cloud-native architecture
- **Maintainability**: Separation of concerns
- **Reliability**: Automated testing and monitoring
- **Performance**: Optimized data access patterns

### Business Benefits
- **Cost Efficiency**: Pay-per-use cloud resources
- **Data Quality**: Automated validation and testing
- **Agility**: Faster development and deployment cycles
- **Compliance**: Audit trails and data lineage

## Timeline & Milestones

### Week 1-2: Foundation
- [ ] BigQuery setup and initial data ingestion
- [ ] Docker containerization
- [ ] Basic dbt project structure

### Week 3-4: Core Development
- [ ] Complete dbt models and tests
- [ ] Refactor Streamlit app
- [ ] Implement data access layer

### Week 5-6: Automation
- [ ] GitHub Actions CI/CD
- [ ] Mage AI orchestration
- [ ] Monitoring and alerting

### Week 7-8: Deployment & Testing
- [ ] Cloud deployment
- [ ] End-to-end testing
- [ ] Performance optimization

## Risk Mitigation

### Technical Risks
- **Data Migration**: Implement gradual migration with fallbacks
- **Performance**: Load testing and optimization
- **Security**: Proper IAM and secret management

### Business Risks
- **Downtime**: Blue-green deployment strategy
- **Cost Overruns**: Budget monitoring and alerts
- **Skill Gap**: Documentation and training materials

## Success Metrics

### Technical Metrics
- Pipeline success rate (>99%)
- Data freshness (< 1 hour delay)
- Query performance (< 5 seconds)
- Test coverage (>90%)

### Business Metrics
- User engagement with new features
- Forecast accuracy improvements
- Development velocity increase
- Operational cost reduction

---

This roadmap provides a comprehensive guide for transforming the stock forecasting tool into a modern analytics engineering platform. Each phase builds upon the previous one, ensuring a systematic and risk-mitigated approach to modernization.
