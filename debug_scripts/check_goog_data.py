from app_modules.bigquery_client import BigQueryClient
import os
from datetime import datetime, timedelta

# Get the service account key path
service_key_path = r"C:\Users\Shane\Desktop\Service Account Keys\swing-ticker-bigquery-8e46b0acdd41.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_key_path

# Create client and query
bq = BigQueryClient()
query = """
SELECT 
  symbol,
  date, 
  close
FROM `swing-ticker-bigquery.stock_data.daily_prices`
WHERE symbol = 'GOOG'
ORDER BY date DESC
"""

result = bq.client.query(query).to_dataframe()
print(f"Total GOOG rows in BigQuery: {len(result)}")
print(f'Date range: {result["date"].min()} to {result["date"].max()}')
print(f'First 5 dates: {result["date"].head().tolist()}')
print(f'Last 5 dates: {result["date"].tail().tolist()}')
