# Stock Forecasting Tool - Modernization Roadmap

<!-- FORMATTING NOTE: This roadmap has two sections only:
1. PHASES: Active work items with consistent Phase X format, objectives, and implementation steps
2. COMPLETED: Finished work grouped by category with checkmarks and brief descriptions
Keep all active work in Phases, all finished work in Completed. No mixing of statuses. -->

## Current Status & Next Recommended Steps

**Top Priority**: Did technical indicators get applied to refreshed data?
**Last Session Focus**: Cloud Function deployment and infrastructure testing
**Current State**: Cloud Function infrastructure COMPLETED - Deployment, timezone handling, BigQuery connectivity all working
**Major Achievement**: Successfully resolved file naming conflicts, timezone datetime issues, and established production-ready Cloud Function deployment pipeline
**Current Issues**: 
1. **Security Risk**: Cloud Function deployed unauthenticated - anyone can trigger expensive operations
2. **Mock Logic**: Cloud Function uses mock optimization instead of real hyperparameter tuning - needs actual optimization implementation

### Next Recommended Steps
- **Security Priority**: Enable Cloud Function authentication to protect against unauthorized access and costs
- **Immediate Priority**: Implement real optimization logic in Cloud Function using existing hyperparameter tuning codebase  
- **Testing**: Verify actual parameter optimization writes to BigQuery optimal_parameters table
- **Integration**: Ensure Cloud Function optimization matches local optimization results
- **Phase 2**: Begin "Performance Optimization & Monitoring" after real optimization is working

## Phases

### Phase 1A: Cloud Function Security & Authentication
**Priority**: HIGH - Security vulnerability with current unauthenticated access
**Objective**: Enable authentication on parameter optimizer Cloud Function to prevent unauthorized access and protect against cost abuse

**Current Security Risk**: 
- ❌ Cloud Function deployed with `--allow-unauthenticated` flag
- ❌ **Anyone with URL can trigger expensive operations** (2GB memory, 1-hour timeout, BigQuery processing)
- ❌ **Potential cost abuse** from malicious actors discovering public endpoint
- ❌ **Business data exposure** through function responses (symbol lists, optimization status)
- ❌ **No access control** for financial data processing function

**Implementation Steps**:

1. **Redeploy with Authentication** (15 minutes):
   ```bash
   cd cloud_functions/parameter_optimizer
   gcloud functions deploy parameter-optimizer \
     --gen2 \
     --runtime python311 \
     --trigger-http \
     --no-allow-unauthenticated \  # Key change - enable authentication
     --memory 2Gi \
     --timeout 3600 \
     --region us-central1 \
     --set-env-vars="BIGQUERY_PROJECT_ID=stock-forecasting-tool-2025,BIGQUERY_DATASET_ID=stock_data,RAW_DATA_TABLE_ID=raw_stock_data,OPTIMAL_PARAMS_TABLE_ID=optimal_parameters"
   ```

2. **Update Testing Method** (10 minutes):
   - Replace unauthenticated PowerShell testing with authenticated requests
   - Use `gcloud auth print-identity-token` for local testing
   - Update testing commands in documentation

3. **Verify Access Control** (5 minutes):
   - Test that unauthenticated requests are now rejected (401/403 responses)
   - Verify authenticated requests work correctly
   - Document authentication requirements

4. **Update Deployment Scripts** (10 minutes):
   - Modify `deploy.bat` and `deploy.sh` to remove `--allow-unauthenticated`
   - Add authentication documentation to README.md
   - Include testing examples with authentication

5. **Future Cloud Scheduler Integration** (for reference):
   - When implementing weekly automation, use service account authentication
   - Configure OIDC token-based authentication for scheduler jobs

**Expected Outcomes**:
- Cloud Function requires authentication for all requests
- Cost protection against unauthorized usage
- Security compliance for financial data processing
- Foundation for proper Cloud Scheduler integration

**Testing Commands (Post-Implementation)**:
```powershell
# Get auth token
$token = gcloud auth print-identity-token

# Test authenticated request
$headers = @{"Authorization"="Bearer $token"}
Invoke-WebRequest -Uri "https://parameter-optimizer-cjudw6alcq-uc.a.run.app" -Method POST -Headers $headers -Body '{"force_run": true}' -ContentType "application/json"
```

### Phase 1B: Real Parameter Optimization Implementation
**Priority**: CRITICAL - Infrastructure ready, needs actual optimization logic
**Objective**: Replace mock optimization in Cloud Function with real hyperparameter tuning using existing codebase

**Current Status**: 
- ✅ Cloud Function deployment infrastructure working (main.py, timezone fixes, BigQuery connectivity)
- ✅ Mock optimization reporting 46 symbols "processed successfully" 
- ❌ No actual optimization occurring - trigger_optimization() method contains simulation code only
- ❌ Zero rows in optimal_parameters table despite "successful" runs

**Implementation Steps**:

1. **Analysis Phase** (0.5-1 hour):
   - Examine existing optimization logic in `archive/` notebooks (2024.12.18 - Model Eval.ipynb, 2024.12.30 - Winsorizing + Optimization.ipynb)
   - Identify hyperparameter tuning patterns used in local optimization runs
   - Map BigQuery schema requirements for optimal_parameters table (changepoint_prior_scale, seasonality_prior_scale, rmse, mae, smape, etc.)

2. **Module Creation** (1-2 hours):
   - Create `optimization_engine.py` module in `cloud_functions/parameter_optimizer/`
   - Extract core Prophet optimization logic from archive notebooks
   - Implement BigQuery data loading for symbol-specific historical data
   - Add parameter validation and error handling for Cloud Function environment

3. **Core Optimization Logic** (2-3 hours):
   ```python
   class CloudParameterOptimizer:
       def __init__(self, bigquery_client, project_id, dataset_id):
           # Initialize BigQuery connections using existing patterns
           
       def optimize_symbol_parameters(self, symbol: str) -> dict:
           # Load historical data for symbol from BigQuery raw_stock_data table
           # Apply your proven hyperparameter tuning methodology:
           #   - Cross-validation approach from 2024.12.18 notebooks
           #   - Parameter grid search (changepoint_prior_scale, seasonality_prior_scale)
           #   - Performance metrics calculation (RMSE, MAE, SMAPE)
           # Return optimized parameters matching BigQuery schema
           
       def batch_optimize_symbols(self, symbols: list) -> list:
           # Process multiple symbols with error handling and logging
           # Store results directly to BigQuery optimal_parameters table
   ```

4. **BigQuery Integration** (1 hour):
   - Implement direct writes to `stock-forecasting-tool-2025.stock_data.optimal_parameters`
   - Follow existing schema: symbol, changepoint_prior_scale, seasonality_prior_scale, rmse, mae, smape, optimization_date, data_points_used, cv_folds, created_at, updated_at
   - Add upsert logic for re-optimization scenarios
   - Include comprehensive logging for debugging

5. **Replace Mock Implementation** (0.5 hour):
   - Update `ParameterOptimizer.trigger_optimization()` in main.py
   - Remove simulation code, import and use CloudParameterOptimizer
   - Maintain same response format but with actual optimization results
   - Preserve error handling and logging patterns

6. **Dependencies & Requirements** (0.5 hour):
   - Add Prophet, scikit-learn, hyperopt to `requirements.txt`
   - Verify Cloud Function memory (2Gi) and timeout (3600s) are sufficient for optimization workload
   - Test package compatibility in Cloud Function environment

7. **Testing & Validation** (1 hour):
   - Deploy updated Cloud Function with real optimization
   - Test single symbol optimization first (e.g., SPY)
   - Verify BigQuery writes are successful and data matches expected schema
   - Run full 46-symbol optimization and confirm population of optimal_parameters table
   - Compare optimization results with local runs for consistency validation

**Expected Outcomes**:
- Cloud Function executes actual hyperparameter tuning using proven methodology from archive notebooks
- BigQuery optimal_parameters table populated with 46 rows of optimized Prophet parameters
- Optimization results match quality and consistency of local optimization runs
- Weekly automated parameter optimization becomes fully functional

**Dependencies Required**:
- Prophet library and dependencies in Cloud Function environment
- Archive notebook methodology (grid search, cross-validation, performance metrics)
- BigQuery write permissions and schema compliance
- Sufficient Cloud Function compute resources (current 2Gi memory, 1 hour timeout should be adequate)

**Risk Mitigation**:
- Start with single symbol testing before full batch processing
- Implement comprehensive error handling and logging throughout optimization pipeline
- Add timeout protection for individual symbol optimization (prevent function timeout)
- Include fallback mechanisms for optimization failures per symbol

### Phase 2: Performance Optimization & Monitoring  
**Priority**: High  
**Objective**: Enhance application performance and implement production monitoring

**Implementation Steps**:
- Implement BigQuery connection pooling for improved query performance
- Add comprehensive error handling with graceful fallbacks for data pipeline
- Create operational dashboards for system health and performance metrics
- Enhance user experience with progress indicators and status updates

### Phase 3: Advanced Feature Engineering
**Priority**: Medium  
**Objective**: Integrate research from archive notebooks and enhance technical analysis capabilities

**Implementation Steps**:
- Extract winsorization techniques from `2024.12.30 - Winsorizing + Optimization` notebooks
- Add dynamic technical indicators (Stochastic, Williams %R, CCI, ATR)
- Implement rolling correlation windows and sector analysis
- Create configurable feature engineering pipeline with outlier detection

### Phase 4: Production Monitoring & Automation
**Priority**: Medium  
**Objective**: Ensure production reliability and operational excellence

**Implementation Steps**:
- Add data pipeline monitoring with freshness alerts and quality validation
- Create automated health checks and alerting systems
- Implement comprehensive logging and error tracking
- Enhance deployment automation and rollback capabilities

### Phase 5: Advanced Analytics Platform
**Priority**: Low  
**Objective**: Transform into comprehensive financial analytics platform

**Implementation Steps**:
- Implement dbt transformation layer for SQL-based feature engineering
- Add real-time data streaming and instant market updates
- Integrate alternative ML models and ensemble methods
- Create portfolio-level analysis and risk management tools

## Completed

### Data Infrastructure
- COMPLETED: **BigQuery Integration**: Complete warehouse with 1.4M+ rows across multiple symbols
- COMPLETED: **Data Orchestration**: Production Google Cloud Functions pipeline with daily automated updates  
- COMPLETED: **Environment Setup**: Dedicated conda environment with resolved dependencies
- COMPLETED: **Data Quality**: Comprehensive validation, deduplication logic, no anomalies detected
- COMPLETED: **Trading Calendar**: Cloud Functions use NYSE calendar for business-day-only updates

### Application Architecture
- COMPLETED: **Modular Design**: 12 specialized modules with clean separation of concerns
- COMPLETED: **Prophet Forecasting**: Hyperparameter optimization with cross-validation
- COMPLETED: **Technical Indicators**: SMA, Bollinger Bands, RSI, MACD, Death Cross, Golden Cross
- COMPLETED: **Market Correlation**: Multi-asset correlation analysis and business KPIs
- COMPLETED: **User Interface**: Streamlit app with BigQuery toggle and source selection

### Production Infrastructure  
- COMPLETED: **Containerization**: Multi-stage Docker builds optimized for production
- COMPLETED: **CI/CD Pipeline**: GitHub Actions with automated testing and security scanning
- COMPLETED: **Code Quality**: 85%+ test coverage, Black formatting, comprehensive linting
- COMPLETED: **Deployment Ready**: Container registry configured for Google Cloud Run

### Data Validation & Diagnostics
- COMPLETED: **BigQuery Diagnostics**: Fixed diagnostic script, verified 1.4M+ clean rows
- COMPLETED: **Deduplication Logic**: Handles systematic duplicates, preserves latest data  
- COMPLETED: **Price Validation**: No NULL values, negative prices, or data corruption
- COMPLETED: **Cross-Signal Implementation**: Death Cross and Golden Cross calculations working

### Research Foundation
- COMPLETED: **Archive Notebooks**: Winsorization, hyperparameter tuning, model evaluation techniques
- COMPLETED: **Optimization Methods**: Bayesian optimization and grid search implementations  
- COMPLETED: **Data Preprocessing**: Sophisticated cleaning and transformation pipelines
- COMPLETED: **Performance Metrics**: Advanced cross-validation and model comparison frameworks

### Symbol-Specific Model Optimization
- COMPLETED: **BigQuery Optimal Parameters Table**: Created `optimal_parameters` table with comprehensive schema for storing symbol-specific Prophet parameters
- COMPLETED: **Background Optimization Script**: Built `optimize_symbol_parameters.py` using grid search methodology from archive notebooks (2024.12.18 - Model Eval.ipynb)  
- COMPLETED: **Parameter Lookup Service**: Implemented `parameter_lookup.py` with fallback mechanisms and service initialization
- COMPLETED: **UI Integration**: Updated `ui_intro.py` to automatically load optimal parameters with user override capability
- COMPLETED: **Pipeline Testing**: Created comprehensive test suite validating table operations, lookups, and end-to-end workflow
- COMPLETED: **Optimization Infrastructure**: Full pipeline ready for production use - faster forecasts with pre-computed optimal parameters per symbol
- COMPLETED: **BigQuery Project Migration**: Successfully migrated from stock-forecasting-tool-prod to stock-forecasting-tool-2025
- COMPLETED: **Codebase References Update**: Updated all hardcoded project IDs and table names throughout codebase
- COMPLETED: **Data Pipeline Verification**: 21,998 rows loaded across 44 symbols, 95.7% success rate
- COMPLETED: **Configuration Standardization**: Updated environment files, Cloud Functions, and CI/CD references

### Cloud Function Enhancement & API Management
- COMPLETED: **API Key Conflict Resolution**: Fixed root cause of Alpha Vantage API key conflicts
- COMPLETED: **Environment Variable Management**: Removed incorrect system-level environment variables, standardized on .env file
- COMPLETED: **Rate Limiter Integration**: Added AVAPIRateLimiter with 75 requests/minute for premium Alpha Vantage tier
- COMPLETED: **Dynamic Symbol Discovery**: GCF now queries BigQuery for existing symbols instead of hardcoded lists

### Parameter Optimization Scheduling Infrastructure
- COMPLETED: **Weekly Optimization Cloud Function**: Created `weekly-parameter-optimizer` with trading day validation and dependency management
- COMPLETED: **Smart Scheduling Logic**: Runs only on last trading day of week with automatic raw data freshness validation
- COMPLETED: **Dependency Management**: Built-in waiting logic (up to 30 minutes) ensures raw data is current before optimization begins  
- COMPLETED: **Multiple Scheduling Options**: Conservative (Fri 11:30 PM), Aggressive (Fri 11:00 PM), and Safe (Sat 12:30 AM) deployment modes
- COMPLETED: **Deployment Automation**: Windows and Linux deployment scripts with Cloud Scheduler job creation
- COMPLETED: **Comprehensive Documentation**: Complete README, troubleshooting guide, and architectural overview
- COMPLETED: **Error Handling**: Robust failure recovery with detailed logging and status reporting
- COMPLETED: **Production Integration**: Seamless integration with existing BigQuery optimal parameters table and Streamlit UI
- COMPLETED: **Schema Standardization**: Updated GCF to use ingested_at column matching BigQuery schema
- COMPLETED: **Deployment Automation**: Enhanced deployment scripts with automatic environment variable handling

### BigQuery Schema Compliance & Pipeline Debugging
- COMPLETED: **BigQuery Schema Issue Resolution**: Fixed "Required field source cannot be null" error in GCF pipeline
- COMPLETED: **Data Pipeline Debugging**: Created comprehensive debugging tool (gcf_data_pipeline_debug.py) for validation
- COMPLETED: **Schema Compliance Verification**: Updated both GCF and bulk loading scripts to include source field
- COMPLETED: **End-to-End Pipeline Validation**: Successfully loaded SPY and GOOG symbols with full schema compliance
- COMPLETED: **Production Pipeline Testing**: Verified data flow from API fetch to BigQuery storage with all required fields

### Streamlit Cloud Authentication Resolution & Production Deployment
- COMPLETED: **Enhanced Authentication Priority**: Updated BigQueryClient to prioritize Streamlit secrets over environment variables
- COMPLETED: **Timeout Protection**: Added 5-second timeout for Google Cloud metadata service to prevent hanging  
- COMPLETED: **JSON Format Validation**: Fixed service account JSON formatting issues in secrets configuration
- COMPLETED: **Diagnostic Tooling**: Created comprehensive authentication diagnostic script for troubleshooting
- COMPLETED: **Streamlit Secrets Integration**: Full integration with st.secrets for seamless cloud deployment
- COMPLETED: **Authentication Testing**: Verified 4/4 authentication tests pass locally with proper BigQuery connectivity
- COMPLETED: **Production Deployment**: Successfully deployed application to Streamlit Cloud with working BigQuery integration
- COMPLETED: **JSON Formatting Resolution**: Created JSON formatter utility to resolve "Invalid control character" errors in production
- COMPLETED: **Production Authentication Verification**: BigQuery authentication fully functional in Streamlit Cloud environment

### Cloud Function Infrastructure & Deployment Resolution
- COMPLETED: **File Naming Conflict Resolution**: Fixed parameter optimizer Cloud Function deployment by renaming optimizer_main.py to main.py as required by Google Cloud Functions
- COMPLETED: **Timezone Handling Fix**: Resolved "can't subtract offset-naive and offset-aware datetimes" error by updating all datetime operations to use timezone-aware UTC timestamps
- COMPLETED: **Environment Variable Configuration**: Fixed BigQuery project ID environment variable parsing issues that were causing malformed connection strings
- COMPLETED: **Production Cloud Function Deployment**: Successfully deployed parameter-optimizer Cloud Function with proper BigQuery connectivity, HTTP triggers, and 2Gi memory/1-hour timeout configuration
- COMPLETED: **Infrastructure Testing**: Verified Cloud Function can connect to BigQuery, retrieve 46 symbols from raw_stock_data table, and execute trading day validation logic
- COMPLETED: **Mock Optimization Verification**: Confirmed Cloud Function infrastructure works end-to-end but currently uses simulation code instead of real optimization logic
