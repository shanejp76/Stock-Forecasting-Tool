# Parameter Optimizer Cloud Function

This Cloud Function performs weekly parameter optimization for Prophet models after raw data updates complete on the last trading day of each week.

## Features

- **Smart Scheduling**: Only runs on the last trading day of the week (typically Friday, but handles holidays)
- **Data Validation**: Ensures raw data is current before starting optimization
- **Dependency Management**: Waits for raw data upsert to complete before optimization
- **Comprehensive Logging**: Detailed logging for monitoring and debugging
- **Flexible Triggers**: Can be triggered manually or automatically via Cloud Scheduler

## Architecture

```
Raw Data Update (Daily 10:00 PM ET)
          ↓
    Last Trading Day?
          ↓ (Yes, Friday)
Parameter Optimization (11:30 PM ET)
          ↓
   Optimal Parameters Updated
```

## Deployment

### Prerequisites

1. Google Cloud CLI installed and authenticated
2. BigQuery tables created (`raw_stock_data`, `optimal_parameters`)
3. Appropriate IAM permissions for BigQuery and Cloud Functions

### Deploy the Function

**Windows:**
```batch
cd cloud_functions\parameter_optimizer
deploy.bat [YOUR_GCP_PROJECT_ID]
```

**Linux/Mac:**
```bash
cd cloud_functions/parameter_optimizer
./deploy.sh [YOUR_GCP_PROJECT_ID]
```

### Cloud Scheduler Configuration

The deployment script automatically creates a Cloud Scheduler job:

- **Schedule**: `30 23 * * FRI` (11:30 PM ET on Fridays)
- **Timing**: 30 minutes after raw data update completes
- **Timezone**: America/New_York

## Manual Testing

Test the function directly:

```bash
curl -X POST "https://REGION-PROJECT.cloudfunctions.net/parameter-optimizer" \
  -H "Content-Type: application/json" \
  -d '{
    "force_run": true,
    "symbols": ["AAPL", "GOOGL"]
  }'
```

## Request Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `force_run` | boolean | Skip trading day validation | `false` |
| `force_reoptimize` | boolean | Re-optimize even if recent params exist | `false` |
| `symbols` | array | Override symbols to optimize | All symbols from raw data |

## Response Format

```json
{
  "status": "success",
  "optimization_result": {
    "status": "completed",
    "symbols_processed": ["AAPL", "GOOGL"],
    "optimization_duration": "15.2 minutes"
  },
  "data_validation": {
    "status": "current",
    "latest_date": "2025-09-21",
    "symbols_count": 5
  },
  "trading_day_info": {
    "is_trading_day": true,
    "is_last_trading_day_of_week": true
  },
  "timestamp": "2025-09-21T23:30:00Z"
}
```

## Error Handling

The function handles various scenarios:

- **Not a Trading Day**: Skips optimization unless `force_run=true`
- **Stale Raw Data**: Waits for fresh data or skips if too old
- **No Symbols**: Gracefully handles empty symbol lists
- **Optimization Failures**: Logs errors and continues with other symbols

## Monitoring

### View Logs
```bash
gcloud functions logs read parameter-optimizer --region=us-central1
```

### Check Cloud Scheduler Jobs
```bash
gcloud scheduler jobs list
```

### Monitor BigQuery Tables
```sql
-- Check optimization status
SELECT symbol, last_optimized, parameters 
FROM `project.stock_data.optimal_parameters` 
ORDER BY last_optimized DESC;
```

## Configuration

Environment variables set during deployment:

- `BIGQUERY_PROJECT_ID`: Your GCP project ID
- `BIGQUERY_DATASET_ID`: Dataset containing tables (default: `stock_data`)
- `RAW_DATA_TABLE_ID`: Raw stock data table (default: `raw_stock_data`)
- `OPTIMAL_PARAMS_TABLE_ID`: Optimal parameters table (default: `optimal_parameters`)

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- `prophet`: Prophet forecasting model
- `google-cloud-bigquery`: BigQuery integration
- `pandas-market-calendars`: Trading day validation
- `hyperopt`: Bayesian optimization
- `scikit-learn`: Cross-validation and metrics

## Integration with Main Application

The optimized parameters are automatically used by your Streamlit application through the `parameter_lookup` module:

1. User selects a symbol in the UI
2. `parameter_lookup.py` queries optimal parameters from BigQuery
3. If parameters exist and are recent, they're used for forecasting
4. Otherwise, default parameters are used as fallback

## Troubleshooting

### Common Issues

1. **"Raw data not current"**: Wait for daily data update or use `force_run=true`
2. **BigQuery permissions**: Ensure Cloud Function service account has BigQuery Editor role
3. **Timeout errors**: Increase function timeout or reduce symbols per batch
4. **Import errors**: Verify all dependencies are in `requirements.txt`

### Debug Mode

For detailed debugging, modify the main function locally:

```python
os.environ["BIGQUERY_PROJECT_ID"] = "your-project"
result = parameter_optimization(mock_request)
print(json.dumps(result, indent=2))
```