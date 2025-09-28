# Stock Forecasting Tool - Modernization Roadmap

<!-- FORMATTING NOTE: This roadmap has two sections only:
1. PHASES: Active work items with consistent Phase X format, objectives, and implementation steps
2. COMPLETED: Finished work grouped by category with checkmarks and brief descriptions
Keep all active work in Phases, all finished work in Completed. No mixing of statuses. -->

## Current Status & Next Recommended Steps

**Last Session Focus**: BigQuery connection debugging and authentication diagnostics  
**Current State**: UI-based debugging attempts NOT VISIBLE - Need server-side diagnostics approach
**Major Issue Identified**: BigQuery client showing "not available/connected" in Streamlit Cloud despite proper credentials
**Authentication Status**: 
1. ✅ **Service Account Configured**: stock-forecasting-sa@stock-forecasting-tool-2025.iam.gserviceaccount.com found in Streamlit secrets
2. ✅ **Project ID Correct**: stock-forecasting-tool-2025 properly configured  
3. ✅ **JSON Credentials Present**: GOOGLE_APPLICATION_CREDENTIALS_JSON exists in secrets
4. ❌ **Connection Failing**: BigQuery client.is_available() returns False in production
5. ❌ **Root Cause Unknown**: Need detailed server-side diagnostics to identify specific failure point

### Next Recommended Steps
- **IMMEDIATE PRIORITY**: Implement enhanced BigQuery client diagnostics with server-side logging
- **Phase 1A**: Create minimal reproduction test for BigQuery connection failures  
- **Phase 1B Priority**: Implement real optimization logic in Cloud Function using existing hyperparameter tuning codebase
- **Testing**: Verify actual parameter optimization writes to BigQuery optimal_parameters table
- **Integration**: Ensure Cloud Function optimization matches local optimization results

## Phases

### Phase 0: BigQuery Connection Diagnostics - IMMEDIATE PRIORITY
**Priority**: CRITICAL - Production BigQuery connectivity failure blocking user experience
**Objective**: Implement comprehensive server-side diagnostics to identify root cause of BigQuery client connection failures in Streamlit Cloud

**Current Problem Analysis**:
- ✅ Credentials properly configured in Streamlit secrets (service account JSON validated)
- ✅ Local environment BigQuery client works perfectly (same credentials)
- ❌ Streamlit Cloud BigQuery client.is_available() returns False consistently  
- ❌ UI-based debugging invisible to users (UI updates not rendering properly)
- ❌ Generic "not available/connected" error provides no actionable information
- ❌ All GOOG and other symbol queries falling back to Alpha Vantage unnecessarily

**Recommended Approach**: Enhanced BigQuery Client + Minimal Reproduction Test

**Implementation Steps**:

1. **Enhanced BigQuery Client Error Reporting** (30-45 minutes):
   ```python
   # Modify app_modules/bigquery_client.py
   class BigQueryClient:
       def __init__(self):
           self._connection_available = False
           self._connection_error_details = None  # NEW: Store specific error info
           self._auth_method_used = None          # NEW: Track which auth method worked/failed
           self._last_test_results = {}           # NEW: Store detailed test results
   
       def test_connection(self) -> bool:
           """Enhanced connection testing with detailed error capture"""
           # Test 1: Credentials loading and validation
           # Test 2: BigQuery client initialization  
           # Test 3: Simple query execution (SELECT 1)
           # Test 4: Project access verification
           # Test 5: Dataset access verification
           # Store specific failure point and error details
           
       def get_detailed_status(self) -> dict:
           """Return comprehensive connection status for debugging"""
           # Return: auth_method, credentials_valid, client_init_success,
           #         query_test_success, specific_error_messages, permissions_status
   ```

2. **Minimal Reproduction Test Script** (15-20 minutes):
   ```python
   # Create: debug_scripts/bigquery_minimal_test.py
   """
   Standalone BigQuery connection test that can be imported by Streamlit app
   Tests each step of authentication and connection process individually
   Logs detailed results to both console and return structured data
   """
   def test_bigquery_connection_steps():
       results = {
           'step_1_secrets_load': False,
           'step_2_credentials_parse': False, 
           'step_3_client_init': False,
           'step_4_simple_query': False,
           'step_5_project_access': False,
           'step_6_dataset_access': False,
           'error_details': {},
           'auth_method_used': None
       }
       # Test each step individually with try/catch and detailed logging
       return results
   ```

3. **Server-Side Logging Integration** (15-20 minutes):
   ```python
   # Modify load_bigquery_data() in data_sources.py to use enhanced client
   def load_bigquery_data(ticker: str, start_date: date) -> Tuple[pd.DataFrame, str]:
       # When BigQuery fails, call enhanced diagnostics
       if not bq_client.is_available():
           detailed_status = bq_client.get_detailed_status()
           # Log to console (visible in Streamlit Cloud logs)
           print(f"BIGQUERY_DIAGNOSTIC: {json.dumps(detailed_status, indent=2)}")
           
           # Also run minimal reproduction test
           test_results = test_bigquery_connection_steps() 
           print(f"BIGQUERY_REPRO_TEST: {json.dumps(test_results, indent=2)}")
           
           # Return enhanced error message for UI
           return pd.DataFrame(), f"BigQuery unavailable: {detailed_status.get('primary_error', 'unknown')}"
   ```

4. **Streamlit Cloud Log Analysis Setup** (10-15 minutes):
   - Add structured logging format for easy parsing
   - Create log monitoring approach for Streamlit Cloud application logs
   - Document how to access and interpret the diagnostic output
   - Add timestamp and session identifiers for tracking

5. **Testing and Validation** (20-30 minutes):
   - Deploy enhanced diagnostics to Streamlit Cloud
   - Trigger BigQuery connection attempts with various symbols  
   - Capture and analyze detailed diagnostic output from server logs
   - Compare local vs Streamlit Cloud diagnostic results
   - Identify specific failure points (authentication, permissions, network, API limits, etc.)

**Expected Outcomes**:
- **Specific Error Identification**: Know exactly why BigQuery client fails (auth vs permissions vs network vs API)
- **Actionable Information**: Clear next steps based on specific failure mode identified
- **Server-Side Visibility**: Detailed diagnostics visible in Streamlit Cloud logs regardless of UI issues
- **Local vs Production Comparison**: Understand environmental differences causing the issue

**Common Failure Scenarios to Test For**:
- Service account permissions missing for BigQuery API
- Streamlit Cloud network restrictions blocking Google API access
- BigQuery API not enabled for the project in production environment
- Rate limiting or quota exhaustion in production vs local
- Credential format issues specific to Streamlit Cloud secrets parsing
- Project ID or dataset access permissions in production service account

**Success Criteria**:
- Diagnostic output clearly identifies root cause of BigQuery connection failure
- GOOG and other symbols successfully load from BigQuery instead of falling back to Alpha Vantage
- Enhanced error messages provide actionable troubleshooting information
- Server-side logging provides comprehensive debugging capability for future issues

**Files to Modify**:
- `app_modules/bigquery_client.py` - Enhanced error reporting and diagnostics
- `app_modules/data_sources.py` - Integration of enhanced diagnostics  
- `debug_scripts/bigquery_minimal_test.py` - New standalone test script
- Deployment to Streamlit Cloud for testing

**Time Estimate**: 1.5-2 hours total implementation + testing

### Phase 1A: Cloud Function Security & Authentication - COMPLETED
**Priority**: HIGH - Security vulnerability resolved
**Objective**: Enabled authentication on parameter optimizer Cloud Function to prevent unauthorized access and protect against cost abuse

**Security Implementation COMPLETED**: 
- ✅ Updated `deploy.bat` and `deploy.sh` with `--no-allow-unauthenticated` flag
- ✅ Eliminated anonymous access to expensive operations (2GB memory, 1-hour timeout, BigQuery processing)
- ✅ Protected against potential cost abuse from malicious actors
- ✅ Secured business data exposure through function responses
- ✅ Implemented proper access control for financial data processing function
- ✅ Updated README.md with authentication requirements and testing examples
- ✅ Added comprehensive troubleshooting guide for authentication issues
- ✅ Verified Cloud Scheduler compatibility (uses service account authentication automatically)

**Testing Commands (Authentication Required)**:
```bash
# Get auth token and test
curl -X POST "https://FUNCTION_URL" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"force_run": true}'
```

**Security Benefits Achieved**:
- Cost protection against unauthorized usage
- Security compliance for financial data processing  
- Audit trail for all authenticated requests
- Foundation for proper Cloud Scheduler integration

**Status**: COMPLETED - Ready for secure redeployment

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

### Cloud Function Security & Authentication
- COMPLETED: **Authentication Implementation**: Enabled secure authentication on parameter optimizer Cloud Function
- COMPLETED: **Security Vulnerability Fix**: Removed --allow-unauthenticated flag from deployment scripts (deploy.bat, deploy.sh)
- COMPLETED: **Cost Protection**: Eliminated anonymous access to expensive operations (2Gi memory, 1-hour timeout, BigQuery processing)
- COMPLETED: **Documentation Update**: Added comprehensive authentication requirements, testing examples, and troubleshooting guide to README.md
- COMPLETED: **Access Control**: Implemented proper authentication for financial data processing function
- COMPLETED: **Testing Verification**: Confirmed gcloud authentication commands work correctly for manual testing
- COMPLETED: **Cloud Scheduler Compatibility**: Verified service account authentication continues to work for automated scheduling
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

### Dependency Management & Docker Security
- COMPLETED: **Dependency Version Pinning**: All core dependencies pinned to exact working versions in requirements.txt (numpy==1.26.4, pandas==2.2.3, google-cloud-bigquery==3.38.0, etc.)
- COMPLETED: **NumPy Compatibility Resolution**: Fixed NumPy 2.x compatibility issues with pandas/numexpr by pinning to 1.26.4
- COMPLETED: **Docker Build Security**: Added pip version pinning (23.2.1) to Dockerfiles for consistent builds across environments
- COMPLETED: **Container Health Checks**: Added curl dependency to Docker images for proper health check functionality
- COMPLETED: **Build Determinism**: Docker containers now use identical dependency versions as development environment
- COMPLETED: **Compatibility Testing**: Verified all pinned versions work together without conflicts in both conda and Docker environments
