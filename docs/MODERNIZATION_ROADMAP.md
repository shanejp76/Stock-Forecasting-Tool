# Stock Forecasting Tool - Modernization Roadmap

## Overview
This document outlines the transformation of the existing Streamlit-based stock forecasting application into a modern, production-ready analytics engineering platform. The modernization will introduce cloud data warehousing, proper data transformation pipelines, and full automation.

## Current Architecture (As of August 2025)
- **Frontend**: Streamlit web application with modular architecture
- **Data Source**: Alpha Vantage and Finnhub APIs via dedicated client modules
- **Processing**: Prophet-based forecasting with comprehensive technical indicators
- **Model Training**: Hyperparameter optimization with cross-validation
- **Features**: Market correlation analysis, business KPIs, winsorization techniques
- **Testing**: Comprehensive test suite with pytest framework
- **Containerization**: Docker multi-stage builds with production optimization
- **CI/CD**: GitHub Actions with automated testing, security scanning, and container publishing
- **Storage**: Local/temporary data handling with caching mechanisms
- **Deployment**: Containerized local execution, ready for cloud deployment

## Target Architecture
- **Data Warehouse**: Google BigQuery for centralized data storage
- **Data Transformation**: dbt (data build tool) for SQL-based transformations
- **Orchestration**: Mage AI for workflow scheduling
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Frontend**: Enhanced Streamlit app querying transformed data
- **Deployment**: Containerized with Docker

## Phase 1: Data Warehouse Integration (HIGH PRIORITY)

### Objectives
- Replace direct API calls with centralized data warehouse
- Implement historical data persistence and efficient querying
- Separate data ingestion from application logic
- Enable data lineage and auditability

### Current Status: PROJECT CONFIGURATION ISSUE
**ISSUE IDENTIFIED** Project name mismatch between BigQuery console access and bulk loader configuration.

### Implementation Steps
1. **COMPLETED: Setup Google Cloud Environment**
   - Installed Google Cloud SDK locally
   - Created BigQuery-enabled GCP project (`stock-forecasting-tool-prod`)
   - Configured authentication and enabled required APIs
   - Created BigQuery dataset with proper schema
   - Tested end-to-end data ingestion and retrieval

2. **COMPLETED: Scale Data Ingestion**
   - DONE: Created BigQuery dataset for stock data storage
   - DONE: Configured service account and authentication  
   - DONE: Designed normalized schema for OHLC data, technical indicators, and metadata
   - DONE: Enhanced bulk loading script with progress tracking, checkpointing, and premium rate limiting
   - DONE: Integrated symbol universe retrieval from Alpha Vantage LISTING_STATUS endpoint
   - DONE: Implemented enterprise-grade data ingestion pipeline

3. **CURRENT ISSUE: Project Access Mismatch**
   - **Problem**: BigQuery console shows `sunlit-apricot-424118-r7` project, but bulk loader configured for `stock-forecasting-tool-prod`
   - **Impact**: Data may not be reaching intended destination
   - **Required Fix**: Update .env configuration to match accessible project OR configure access to intended project
   - **Reset Needed**: Clear `data/bulk_load_progress.pkl` checkpoint to restart with correct configuration

3. **READY: Initial Data Loading Strategy**
   - **One-time bulk load**: 2 years historical data per symbol
   - **Manual operations**: Basic data loading and querying for Phase 1
   - **Schema validation**: Ensure data structure matches application needs
   - **Foundation testing**: Verify BigQuery integration before automation

4. **Application Integration**
   - Modify `data_handler.py` to query BigQuery instead of direct API calls
   - Maintain API fallback for testing and edge cases
   - Test end-to-end functionality with BigQuery data source
   - Validate forecasting models work with warehouse data

3. **Raw Data Schema**
   ```sql
   CREATE TABLE raw_stock_data (
     symbol STRING,
     date DATE,
     open FLOAT64,
     high FLOAT64,
     low FLOAT64,
     close FLOAT64,
     adjusted_close FLOAT64,
     volume INT64,
     dividend FLOAT64,
     split FLOAT64,
     ingested_at TIMESTAMP,
     source STRING
   );
   
   CREATE TABLE technical_indicators (
     symbol STRING,
     date DATE,
     sma_20 FLOAT64,
     sma_50 FLOAT64,
     bollinger_upper FLOAT64,
     bollinger_lower FLOAT64,
     rsi FLOAT64,
     macd FLOAT64,
     signal FLOAT64,
     calculated_at TIMESTAMP
   );
   ```

4. **Application Integration**
   - Modify existing `data_handler.py` to query BigQuery instead of APIs
   - Maintain API fallback for testing and edge cases
   - Test end-to-end functionality with BigQuery data source
   - Validate forecasting models work with warehouse data

### Priority: CRITICAL
This phase unlocks historical analysis, reduces API dependencies, and enables advanced analytics.

### Implementation Note: Foundation Before Automation
Phase 1 focuses on basic BigQuery integration with manual operations. The incremental update automation (daily append/delete) is deferred to **Phase 5: Orchestration & Automation** to avoid premature optimization. This allows validation of the core architecture before adding scheduling complexity.

### Development Approach: Native Python First
During Phase 1 implementation, use direct Python development for faster iteration:
```bash
pip install -r requirements.txt
streamlit run main.py
```
Local Docker testing can be deferred until Phase 2 when production simulation becomes valuable for BigQuery performance validation.

## Phase 2: Advanced Feature Engineering (MEDIUM PRIORITY)

### Objectives
- Integrate winsorization techniques from research notebooks
- Implement dynamic technical indicators
- Add advanced market correlation features
- Create feature selection and importance analysis

### Current Status: PARTIALLY IMPLEMENTED
Research in archive notebooks shows promising winsorization and optimization techniques that need integration.

### Implementation Steps
1. **Winsorization Integration**
   - Extract dynamic winsorization functions from `2024.12.30 - Winsorizing + Optimization` notebooks
   - Add to `data_pipeline.py` as configurable preprocessing step
   - Implement rolling window outlier detection and correction

2. **Enhanced Technical Indicators**
   - Expand beyond current SMA, Bollinger Bands, RSI, MACD
   - Add momentum indicators (Stochastic, Williams %R, CCI)
   - Implement volatility measures (ATR, Volatility Cones)
   - Create composite indicators and factor scores

3. **Market Correlation Enhancements**
   - Extend current `market_correlation.py` functionality
   - Add sector correlation analysis
   - Implement rolling correlation windows
   - Create correlation-based risk metrics

4. **Feature Engineering Pipeline**
   ```python
   # Example implementation structure
   class FeatureEngineer:
       def __init__(self, winsorization_params, indicator_config):
           self.winsorization = DynamicWinsorizer(**winsorization_params)
           self.indicators = TechnicalIndicators(**indicator_config)
       
       def transform(self, data):
           # Apply winsorization
           data = self.winsorization.transform(data)
           # Generate indicators
           data = self.indicators.calculate_all(data)
           # Create interaction features
           data = self.create_interaction_features(data)
           return data
   ```

### Priority: MEDIUM
These enhancements will improve model accuracy and provide more sophisticated analysis capabilities.

## Phase 3: dbt Data Transformation Layer (LOW PRIORITY)

### Objectives
- Move complex feature engineering from Python to SQL
- Implement version-controlled transformation logic
- Create clean, documented data models for advanced analytics

### Current Status: PLANNED
This phase can be deferred until data volume and complexity justify SQL-based transformations.

### Implementation Steps
1. **dbt Project Setup**
   - Initialize dbt project structure
   - Configure BigQuery connection
   - Set up development and production environments

2. **Data Models**
   - **Staging Models**: Clean and standardize raw BigQuery data
   - **Intermediate Models**: Technical indicators and feature engineering
   - **Mart Models**: Final datasets optimized for ML model training

3. **Advanced SQL Transformations**
   ```sql
   {{ config(materialized='table') }}
   
   WITH base_data AS (
     SELECT *,
       LAG(close, 1) OVER (PARTITION BY symbol ORDER BY date) as prev_close,
       PERCENTILE_CONT(close, 0.05) OVER (
         PARTITION BY symbol 
         ORDER BY date 
         ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
       ) as rolling_5th_percentile
     FROM {{ ref('raw_stock_data') }}
   ),
   
   winsorized_data AS (
     SELECT *,
       CASE 
         WHEN close < rolling_5th_percentile THEN rolling_5th_percentile
         WHEN close > rolling_95th_percentile THEN rolling_95th_percentile
         ELSE close
       END as winsorized_close
     FROM base_data
   )
   
   SELECT * FROM winsorized_data
   ```

### Priority: LOW
Current Python-based approach is sufficient for current data volumes and use cases.

## Phase 4: Application Performance Optimization (MEDIUM PRIORITY)

### Objectives
- Optimize Streamlit app performance and user experience
- Implement advanced caching strategies
- Add real-time features and enhanced analytics
- Improve model training efficiency

### Current Status: PARTIALLY IMPLEMENTED
Basic caching exists, but performance can be significantly improved.

### Implementation Steps
1. **Enhanced Database Connection Layer**
   - Implement connection pooling for BigQuery
   - Add intelligent caching with TTL and invalidation strategies
   - Create data access layer abstracting storage details

2. **Performance Optimizations**
   - Implement lazy loading for expensive computations
   - Add background model training with progress indicators
   - Cache model results with parameter-based invalidation
   - Optimize Prophet model initialization and training

3. **Advanced UI Features**
   - Real-time streaming data updates
   - Interactive parameter tuning with live preview
   - Model comparison dashboard
   - Historical backtesting interface
   - Portfolio-level analysis capabilities

4. **Configuration Management Enhancement**
   ```python
   # Enhanced config structure
   class AppConfig:
       def __init__(self):
           self.api_config = APIConfig()
           self.model_config = ModelConfig()
           self.cache_config = CacheConfig()
           self.ui_config = UIConfig()
       
       @classmethod
       def from_environment(cls, env='production'):
           # Load environment-specific configurations
           pass
   ```

### Priority: MEDIUM
These improvements will significantly enhance user experience and application scalability.

## Phase 4.5: Production Error Handling & Monitoring (HIGH PRIORITY)

### Objectives
- Implement comprehensive error handling for BigQuery-based architecture
- Add robust monitoring and alerting for data pipelines
- Enhance production reliability and user experience
- Create operational dashboards for system health

### Current Status: IMPLEMENT AFTER BIGQUERY MIGRATION
This phase should be implemented once the BigQuery data architecture is stable and the API-direct approach is deprecated.

### Implementation Steps
1. **BigQuery Error Handling**
   ```python
   class DataWarehouseError(Exception):
       """Custom exception for BigQuery/data warehouse errors"""
       pass
   
   class BigQueryClient:
       def _execute_query(self, query, params=None):
           # Comprehensive error handling for:
           # - Connection failures
           # - Query timeouts
           # - Permission errors
           # - Data quality issues
           # - Cost/quota limits
           pass
   ```

2. **Data Pipeline Monitoring**
   - Data freshness alerts (if data is older than X hours)
   - Data quality validation checks
   - Pipeline failure notifications
   - Cost monitoring and budget alerts

3. **Application Error Handling**
   - Graceful fallback when BigQuery is unavailable
   - User-friendly error messages with context
   - Automatic retry logic for transient failures
   - Performance degradation alerts

4. **Operational Dashboards**
   - System health metrics (uptime, response times)
   - Data pipeline status and execution logs
   - User engagement and usage patterns
   - Cost tracking and optimization recommendations

### Benefits
- **Production Reliability**: Robust error handling for cloud infrastructure
- **Proactive Monitoring**: Issues detected before users notice
- **Cost Management**: Prevent runaway BigQuery costs
- **Better User Experience**: Clear guidance when data issues occur

### Priority: HIGH (after BigQuery implementation)
This ensures production readiness of the final cloud-native architecture.

## Docker Deployment Strategy

### Local Development vs Production Simulation Timeline

**Current Phase (BigQuery Integration): Skip Local Docker**
- Focus on native Python development for faster iteration
- Use direct database connections for debugging BigQuery integration
- Leverage existing GitHub Actions CI/CD for container validation

**Phase 2+ (Production Simulation Required): Local Docker Critical**
When these scenarios become important:

1. **BigQuery Performance Testing**
   ```bash
   # Simulate production workloads locally
   docker-compose up -d
   # Test connection pooling under concurrent load
   # Validate query optimization and caching strategies
   # Debug memory usage patterns with large datasets
   ```

2. **Multi-Service Integration Testing**
   ```yaml
   # docker-compose.yml - Future enhanced setup
   services:
     app:
       build: .
       environment:
         - BIGQUERY_PROJECT_ID=${BIGQUERY_PROJECT_ID}
     
     monitoring:
       image: prom/prometheus
       
     load-test:
       image: locustio/locust
       volumes:
         - ./tests/load:/mnt/locust
   ```

3. **Production Error Scenario Testing**
   - Network failures to BigQuery
   - API rate limit scenarios
   - Memory constraints under load
   - Service account authentication issues

### Local Docker Testing Triggers
- **Before BigQuery production deployment**: Validate end-to-end data pipeline
- **Before performance optimization**: Test caching and connection pooling
- **Before multi-service orchestration**: Validate service interactions

### Current Recommendation: Defer Until Phase 2
The application architecture is already validated through GitHub Actions. Local Docker testing becomes valuable when simulating BigQuery performance characteristics and multi-service interactions.

## Phase 5: Orchestration & Automation (HIGH PRIORITY)

### Objectives
- Automate data ingestion and model retraining pipelines
- Implement comprehensive monitoring and alerting
- Schedule regular updates and maintenance tasks
- Create operational dashboards

### Current Status: BASIC AUTOMATION
CI/CD exists for application deployment, but data and ML pipelines need automation.

### Implementation Steps
1. **Incremental Data Pipeline Automation - Trading Days Only**
   - **Daily rolling window maintenance**: Append new day + delete oldest day
   - **Trading calendar integration**: Only execute on NYSE/NASDAQ trading days
   - **Weekend/holiday handling**: Skip processing on non-trading days
   - **Cost optimization**: Process only 2 days worth of data daily vs full refresh
   - **Scheduled jobs**: Cloud Functions or cron for automated execution
   - **Data quality monitoring**: Automated alerts for data freshness and completeness
   - **Retry logic**: Error handling for API failures and BigQuery timeouts

2. **Data Ingestion Orchestration with Trading Calendar**
   ```python
   # Daily incremental update process - trading days only
   import pandas_market_calendars as mcal
   
   class DailyDataMaintenance:
       def __init__(self):
           self.nyse = mcal.get_calendar('NYSE')
           
       def is_trading_day(self, date):
           """Check if date is a NYSE trading day"""
           schedule = self.nyse.schedule(start_date=date, end_date=date)
           return not schedule.empty
           
       def maintain_rolling_window(self, symbol: str):
           # Only execute on trading days
           if not self.is_trading_day(datetime.now().date()):
               return "Skipped: Not a trading day"
               
           # 1. Fetch latest trading day data
           new_data = self.fetch_latest_data(symbol)
           
           # 2. Append new data
           self.append_to_bigquery(new_data, symbol)
           
           # 3. Remove oldest data to maintain 2-year window
           self.remove_oldest_data(symbol)
   ```

3. **ML Pipeline Automation**
   - Schedule weekly model retraining with new data
   - Implement automated model validation and A/B testing
   - Create model performance drift detection
   - Automate hyperparameter optimization experiments

4. **Monitoring & Alerting Framework**
   ```python
   # Example monitoring structure
   class SystemMonitor:
       def check_data_freshness(self):
           # Alert if data is stale
           pass
       
       def check_model_performance(self):
           # Alert if accuracy drops below threshold
           pass
       
       def check_api_health(self):
           # Monitor API availability and rate limits
           pass
   ```

5. **Operational Dashboards**
   - System health and performance metrics
   - Data pipeline status and lineage
   - Model performance tracking over time
   - Business metrics and KPI trends

### Priority: HIGH
Automation is critical for production reliability and scalability.

## Current Status: Production-Ready Infrastructure Complete

### Containerization & CI/CD Pipeline Status [COMPLETED]
The application infrastructure is fully production-ready:
- **Multi-stage Docker builds**: Optimized for both development and production
- **Comprehensive testing**: Unit tests, integration tests, and smoke tests
- **Security scanning**: Automated vulnerability detection with Trivy
- **Code quality enforcement**: Black formatting, isort imports, flake8 linting
- **Automated deployment**: GitHub Actions with staging and production environments
- **Container registry**: Automated image building and publishing

### Application Architecture Status [COMPLETED]
The core application has been fully modularized and optimized:
- **Modular design**: Clean separation of concerns across 12 specialized modules
- **Advanced forecasting**: Prophet models with hyperparameter optimization
- **Technical analysis**: Comprehensive indicators (SMA, Bollinger, RSI, MACD)
- **Market correlation**: Multi-asset correlation analysis
- **Business KPIs**: Financial metrics and performance indicators
- **Robust testing**: 85%+ test coverage with comprehensive test suite

### Research & Development Status [NEEDS INTEGRATION]
Significant advanced features have been researched but need production integration:
- **Winsorization techniques**: Dynamic outlier handling from archive notebooks
- **Hyperparameter optimization**: Bayesian optimization methods developed
- **Model evaluation**: Advanced cross-validation and performance metrics
- **Data preprocessing**: Sophisticated cleaning and transformation pipelines

### Immediate Next Steps

#### 1. Cloud Infrastructure Setup (READY FOR DEPLOYMENT)
The application is containerized and ready for Google Cloud deployment:

```bash
# Configure GCP project
gcloud projects create stock-forecasting-prod
gcloud config set project stock-forecasting-prod

# Enable required services
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Create service account for CI/CD
gcloud iam service-accounts create github-actions

# Deploy to Cloud Run
gcloud run deploy stock-forecasting \
  --image gcr.io/stock-forecasting-prod/stock-forecasting-app:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### 2. Data Infrastructure Priority (CRITICAL)
While the application is production-ready, data infrastructure is the bottleneck:
- Current: Real-time API calls limiting historical analysis
- Needed: BigQuery data warehouse for persistent storage
- Impact: Unlocks advanced analytics and reduces API dependencies

#### 3. Feature Integration Priority (HIGH VALUE)
Archive notebooks contain valuable optimizations ready for integration:
- Dynamic winsorization for improved data quality
- Advanced hyperparameter tuning for better model performance
- Enhanced cross-validation for more robust predictions

## Technology Stack (Current & Target)

### Production Infrastructure [IMPLEMENTED]
- **Containerization**: Docker with multi-stage builds
- **CI/CD**: GitHub Actions with automated testing and deployment
- **Code Quality**: Black, isort, flake8, pytest with 85%+ coverage
- **Security**: Trivy vulnerability scanning, secrets management
- **Deployment**: Ready for Google Cloud Run, container registry configured

### Application Stack [IMPLEMENTED]
- **Frontend**: Streamlit with modular component architecture
- **Forecasting**: Prophet with automated hyperparameter optimization
- **Data Processing**: Pandas, NumPy with technical analysis (ta library)
- **APIs**: Alpha Vantage (OHLC data), Finnhub (symbol metadata)
- **Caching**: Streamlit native caching with decorator optimization

### Research & Development [NEEDS INTEGRATION]
- **Data Quality**: Dynamic winsorization and outlier detection
- **Optimization**: Bayesian hyperparameter tuning, grid search
- **Validation**: Advanced cross-validation with time series awareness
- **Feature Engineering**: Advanced technical indicators and correlation analysis

### Target Infrastructure [PLANNED]
- **Data Warehouse**: Google BigQuery for historical data storage
- **Orchestration**: Cloud Scheduler for automated data ingestion
- **Monitoring**: Cloud Monitoring with custom metrics and alerting
- **Scaling**: Cloud Run with auto-scaling based on demand

## Implementation Priority Matrix

### Critical Path (Complete These First)
1. **BigQuery Data Warehouse** - Unlocks historical analysis and reduces API limitations
2. **Research Integration** - Integrate proven winsorization and optimization techniques
3. **Production Error Handling** - Implement robust monitoring for BigQuery-based architecture
4. **Automation Pipeline** - Schedule data updates and model retraining

### High Value (Significant Impact)
1. **Performance Optimization** - Caching, lazy loading, background processing
2. **Advanced Analytics** - Portfolio analysis, sector correlation, risk metrics
3. **Operational Monitoring** - Health checks, performance tracking, alerts

### Nice to Have (Future Enhancements)
1. **dbt Transformations** - SQL-based feature engineering (when data volume justifies)
2. **Real-time Features** - Live data streaming, instant updates
3. **Advanced ML** - Alternative models, ensemble methods, deep learning

## Updated Timeline & Milestones

### Phase 1: Foundation (Weeks 1-2) - COMPLETED
- **Containerization**: Docker setup and optimization
- **CI/CD Pipeline**: Automated testing and deployment
- **Code Quality**: Linting, formatting, security scanning
- **Modular Architecture**: Clean separation of concerns

### Phase 2: Data Infrastructure (Weeks 3-4) - HIGH PRIORITY
- **BigQuery Setup**: Data warehouse configuration
- **Data Ingestion**: Automated API data collection
- **Historical Storage**: Persistent data with efficient querying
- **Research Integration**: Winsorization and optimization features

### Phase 3: Production Deployment (Weeks 5-6) - READY NOW
- **Cloud Deployment**: Google Cloud Run setup (can be done immediately)
- **Monitoring Setup**: Health checks and performance tracking
- **Security Hardening**: Secrets management and access controls

### Phase 4: Advanced Features (Weeks 7-8) - HIGH VALUE
- **Performance Optimization**: Enhanced caching and background processing
- **Advanced Analytics**: Portfolio analysis and risk metrics
- **Production Error Handling**: Comprehensive monitoring for BigQuery architecture
- **Operational Excellence**: Comprehensive monitoring and alerting

## Benefits Realized & Expected

### Already Achieved
- **Development Velocity**: Modular architecture enables faster feature development
- **Code Quality**: Automated testing and linting prevent production issues
- **Deployment Reliability**: Containerization ensures consistent environments
- **Maintainability**: Clean separation of concerns and comprehensive documentation
- **Security**: Automated vulnerability scanning and secrets management

### Expected from Next Phases
- **Cost Efficiency**: Reduced API costs through data warehousing and caching
- **Performance**: Sub-second response times with persistent data storage
- **Scalability**: Handle multiple users and larger datasets efficiently
- **Data Quality**: Automated validation and outlier detection
- **Business Intelligence**: Historical trend analysis and advanced forecasting

## Risk Assessment & Mitigation

### Technical Risks
- **API Rate Limits**: Mitigated by BigQuery data warehouse implementation
- **Performance Degradation**: Addressed through comprehensive caching strategy
- **Data Quality Issues**: Handled by winsorization and validation pipelines
- **Security Vulnerabilities**: Prevented by automated scanning and secret management

### Business Risks
- **Service Downtime**: Minimized through health checks and auto-scaling
- **Cost Overruns**: Controlled through BigQuery cost monitoring and alerts
- **User Experience**: Enhanced through performance optimization and responsive design

## Success Metrics (Current vs Target)

### Technical Performance
- **Current**: 5-10 second load times for new forecasts
- **Target**: <2 seconds with BigQuery caching
- **Current**: Real-time API dependency for each session
- **Target**: Historical data with real-time updates only when needed

### Business Impact
- **Current**: Single session analysis limited by API calls
- **Target**: Multi-timeframe analysis with historical context
- **Current**: Basic technical indicators
- **Target**: Advanced risk metrics and portfolio-level insights

### Development Efficiency
- **Current**: Manual deployment and testing
- **Target**: Fully automated CI/CD with zero-downtime deployments

---

## Executive Summary

The Stock Forecasting Tool has successfully completed its modernization foundation with production-ready infrastructure, comprehensive testing, and automated deployment capabilities. The application features a sophisticated modular architecture with advanced forecasting capabilities and is ready for immediate cloud deployment.

**Key Achievements:**
- Production-ready containerized application with 85%+ test coverage
- Automated CI/CD pipeline with security scanning and quality enforcement
- Advanced Prophet-based forecasting with hyperparameter optimization
- Comprehensive technical analysis and market correlation features

**Critical Next Steps:**
1. **Data Infrastructure** (Highest Priority): Implement BigQuery warehouse to unlock historical analysis
2. **Research Integration** (High Value): Integrate proven optimization techniques from archive notebooks
3. **Cloud Deployment** (Ready Now): Deploy to Google Cloud Run for production access

**Timeline to Full Production**: 4-6 weeks for complete modernization, with cloud deployment possible immediately.

This roadmap provides a clear path from the current sophisticated application to a fully scalable, cloud-native analytics platform optimized for production use.
