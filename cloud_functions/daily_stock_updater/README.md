# Daily Stock Data Updater - Cloud Function

This Cloud Function provides automated daily updates of stock data to BigQuery using the Alpha Vantage API.

## Features

- **Trading Day Detection**: Only runs on actual trading days using NYSE calendar
- **API Integration**: Fetches data from Alpha Vantage with error handling and rate limiting
- **BigQuery Management**: Automatic table creation, upsert operations, and data cleanup
- **Rolling Window**: Maintains configurable data retention period (default: 3 years)
- **Monitoring**: Comprehensive logging and error reporting
- **Flexible Triggering**: Can be scheduled or triggered manually

## Environment Variables

Set these in your Cloud Function configuration:

```bash
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
BIGQUERY_PROJECT_ID=your-gcp-project-id
BIGQUERY_DATASET_ID=stock_data  # Optional, defaults to 'stock_data'
BIGQUERY_TABLE_ID=raw_stock_data  # Optional, defaults to 'raw_stock_data'
MAX_TRADING_DAYS=500           # Optional, defaults to 500 trading days
```

## BigQuery Table Schema

The function creates a table with this schema:

```sql
CREATE TABLE `project.dataset.raw_stock_data` (
  date DATE NOT NULL,
  symbol STRING NOT NULL,
  open FLOAT64 NOT NULL,
  high FLOAT64 NOT NULL,
  low FLOAT64 NOT NULL,
  close FLOAT64 NOT NULL,
  volume INT64 NOT NULL,
  updated_at TIMESTAMP NOT NULL
);
```

## Usage

### 1. Scheduled Updates (Recommended)

Set up Cloud Scheduler to trigger daily at market close:

```bash
# Create a Cloud Scheduler job
gcloud scheduler jobs create http daily-stock-update \
  --schedule="0 22 * * MON-FRI" \
  --uri="https://your-region-your-project.cloudfunctions.net/daily-stock-update" \
  --http-method=POST \
  --time-zone="America/New_York"
```

### 2. Manual Trigger

Send HTTP POST request with optional parameters:

```bash
curl -X POST https://your-region-your-project.cloudfunctions.net/daily-stock-update \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "force_update": true,
    "outputsize": "full"
  }'
```

### 3. Default Behavior

Without parameters, updates these symbols: AAPL, GOOGL, MSFT, AMZN, TSLA, SPY, QQQ, VTI

## Response Format

```json
{
  "status": "completed",
  "successful_updates": ["AAPL", "GOOGL"],
  "failed_updates": [],
  "total_symbols": 2,
  "success_count": 2,
  "failure_count": 0,
  "timestamp": "2024-01-15T22:05:30.123456"
}
```

## Deployment

1. **Local Testing** (optional):
   ```bash
   pip install -r requirements.txt
   functions-framework --target=daily_stock_update --debug
   ```

2. **Deploy to Cloud Functions**:
   ```bash
   gcloud functions deploy daily-stock-update \
     --runtime python311 \
     --trigger-http \
     --allow-unauthenticated \
     --set-env-vars ALPHA_VANTAGE_API_KEY=your_key,BIGQUERY_PROJECT_ID=your_project \
     --memory 512MB \
     --timeout 540s
   ```

## Error Handling

- **API Rate Limits**: Handles Alpha Vantage rate limiting gracefully
- **Network Issues**: Retries and logs network failures
- **BigQuery Errors**: Reports table creation and data insertion issues
- **Trading Day Validation**: Skips updates on non-trading days unless forced

## Monitoring

The function logs all operations and provides detailed error messages. Key log events:

- Trading day validation results
- API fetch success/failure for each symbol
- BigQuery upsert operations
- Data cleanup operations
- Overall execution summary

## Cost Considerations

- **Alpha Vantage**: Free tier allows 5 API calls per minute, 500 per day
- **Cloud Functions**: Minimal cost for daily 5-minute executions
- **BigQuery**: Storage and query costs based on data volume
- **Cloud Scheduler**: Free tier covers daily scheduling

## Integration with Streamlit App

This function populates the BigQuery tables that your Streamlit app reads from. Once deployed and scheduled, your app will always have fresh data without manual intervention.
