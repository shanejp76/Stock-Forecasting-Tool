# Parameter Optimization Scheduling Solution

## Overview

Your parameter optimization is now configured to run automatically after raw data updates complete on the last trading day of every week. This ensures optimal timing and data freshness for model parameter tuning.

## Complete Architecture

```
Monday-Thursday: Daily Stock Updates (10:00 PM ET)
                      ↓ (Ingest raw data only)
                 
Friday: Daily Stock Update (10:00 PM ET)
                      ↓ (Raw data for Friday)
        Parameter Optimization Trigger (11:30 PM ET)*
                      ↓ (Validates fresh data)
          Weekly Parameter Optimization
                      ↓ (Updates optimal parameters)
         Ready for Next Week's Trading
```

*Timing depends on your chosen scheduling option

## What's Been Implemented

### 1. Enhanced Parameter Optimizer Cloud Function
- **Location**: `cloud_functions/parameter_optimizer/main.py`
- **Features**:
  - Trading day validation (only runs on last trading day of week)
  - Raw data freshness validation with waiting logic
  - Dependency management (waits up to 30 minutes for fresh data)
  - Flexible triggering (manual override capabilities)
  - Comprehensive error handling and logging

### 2. Deployment Infrastructure
- **Windows Script**: `cloud_functions/parameter_optimizer/deploy.bat`
- **Linux Script**: `cloud_functions/parameter_optimizer/deploy.sh`
- **Features**:
  - Automated Cloud Function deployment
  - Multiple scheduling options (Conservative/Aggressive/Safe)
  - Optional backup scheduler creation
  - Environment variable configuration

### 3. Scheduling Options

| Option | Schedule | Description | Use Case |
|--------|----------|-------------|----------|
| **Conservative** | Fri 11:30 PM ET | 30 min after raw data | Guaranteed fresh data |
| **Aggressive** | Fri 11:00 PM ET | Same time as raw data | Uses waiting logic |
| **Safe** | Sat 12:30 AM ET | 1.5 hours after raw data | Maximum reliability |

### 4. Dependency Management
- **Data Validation**: Checks if raw data is current (from last trading day)
- **Waiting Logic**: Waits up to 30 minutes for raw data to be fresh
- **Fallback Options**: Graceful handling if data isn't ready
- **Force Override**: Manual trigger capability bypasses all checks

## Deployment Instructions

### Prerequisites
1. Google Cloud CLI installed and authenticated
2. BigQuery tables already created and populated ✓
3. Appropriate IAM permissions

### Deploy the Parameter Optimizer

**Option 1: Windows**
```batch
cd cloud_functions\parameter_optimizer
deploy.bat YOUR_GCP_PROJECT_ID
```

**Option 2: Linux/Mac**
```bash
cd cloud_functions/parameter_optimizer
./deploy.sh YOUR_GCP_PROJECT_ID
```

### Choose Your Scheduling Strategy
During deployment, you'll be prompted to choose:

1. **Conservative (Recommended)**: Fri 11:30 PM ET
   - Guarantees raw data is complete
   - Most reliable option
   - 30-minute buffer after raw data update

2. **Aggressive**: Fri 11:00 PM ET  
   - Runs simultaneously with raw data update
   - Uses built-in waiting logic
   - Faster execution if raw data completes quickly

3. **Safe**: Sat 12:30 AM ET
   - Maximum time buffer (1.5 hours)
   - Uses `force_run=true` to bypass all checks
   - Best for critical production environments

## Testing & Validation

### Test the Complete Workflow
```bash
python cloud_functions\parameter_optimizer\test_workflow.py --project YOUR_PROJECT_ID
```

### Manual Testing
```bash
# Test with force run
curl -X POST "https://REGION-PROJECT.cloudfunctions.net/parameter-optimizer" \
  -H "Content-Type: application/json" \
  -d '{"force_run": true, "symbols": ["AAPL"]}'

# Test normal scheduling logic
curl -X POST "https://REGION-PROJECT.cloudfunctions.net/parameter-optimizer" \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Monitoring & Maintenance

### View Function Logs
```bash
gcloud functions logs read parameter-optimizer --region=us-central1 --limit=50
```

### Check Scheduler Status
```bash
gcloud scheduler jobs list
gcloud scheduler jobs describe weekly-parameter-optimization-job
```

### Monitor BigQuery Tables
```sql
-- Check recent optimizations
SELECT symbol, last_optimized, parameters 
FROM `project.stock_data.optimal_parameters` 
WHERE last_optimized >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
ORDER BY last_optimized DESC;

-- Check raw data freshness
SELECT MAX(date) as latest_date, MAX(ingested_at) as latest_ingestion
FROM `project.stock_data.raw_stock_data`;
```

## Integration with Your Streamlit App

The optimized parameters automatically integrate with your existing Streamlit app:

1. **Parameter Lookup**: Your app uses `parameter_lookup.py` to get optimal parameters
2. **Fallback Logic**: If no optimized parameters exist, defaults are used
3. **UI Indicators**: Shows optimization status and allows manual override
4. **Real-time Updates**: Fresh parameters are available immediately after optimization

## Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "Raw data not current" | Data update still running | Wait or use `force_run=true` |
| "Not last trading day" | Running on wrong day | Use `force_run=true` or wait for Friday |
| Timeout errors | Large symbol set | Reduce batch size or increase timeout |
| Permission errors | Missing BigQuery access | Check Cloud Function service account IAM |

### Debug Mode
For detailed debugging, check the function logs or run locally:
```python
import os
os.environ["BIGQUERY_PROJECT_ID"] = "your-project"
# Test specific scenarios
```

## Benefits

### For Your Streamlit Application
- **Faster Forecasts**: Optimized parameters reduce prediction time
- **Better Accuracy**: Symbol-specific tuning improves model performance  
- **Automatic Updates**: Parameters refresh weekly without manual intervention
- **Reliability**: Built-in dependency management ensures data consistency

### For Operations
- **Scheduled Automation**: Runs automatically every week
- **Dependency Awareness**: Waits for raw data to be ready
- **Error Recovery**: Graceful handling of various failure scenarios
- **Monitoring**: Comprehensive logging and status reporting

## Next Steps

1. **Deploy**: Run the deployment script with your project ID
2. **Test**: Validate with a manual trigger first
3. **Monitor**: Check logs after the first scheduled run
4. **Optimize**: Adjust scheduling based on your specific needs

Your parameter optimization is now ready to run automatically every week, ensuring your forecasting models stay optimally tuned! 🚀