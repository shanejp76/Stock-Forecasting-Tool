from app_modules.data_handler import load_bigquery_data, process_stock_data
from datetime import date, timedelta
import pandas as pd

# Get the data and see what's happening
ticker = "GOOG"
start_date = date.today() - timedelta(days=2 * 365)  # 2023-09-02

data, source = load_bigquery_data(ticker, start_date)
print(f"After load_bigquery_data: {len(data)} rows")
print(f'Date range: {data["Date"].min().date()} to {data["Date"].max().date()}')

# Check dates around the filter point
start_date_ts = pd.to_datetime(start_date)
before_filter = data[data["Date"] < start_date_ts]
print(f"Rows before {start_date}: {len(before_filter)}")
if len(before_filter) > 0:
    print(f'Dates being filtered out: {before_filter["Date"].dt.date.tolist()}')

after_filter = data[data["Date"] >= start_date_ts]
print(f"Rows after {start_date}: {len(after_filter)}")

# Show what process_stock_data does step by step
print(f"\nprocess_stock_data steps:")
print(
    f'1. Original data order (first 3 dates): {data["Date"].head(3).dt.date.tolist()}'
)
data_reversed = data[::-1].reset_index(drop=True)
print(
    f'2. After reversing (first 3 dates): {data_reversed["Date"].head(3).dt.date.tolist()}'
)
filtered = data_reversed[data_reversed["Date"] >= start_date_ts].reset_index(drop=True)
print(f"3. After filtering: {len(filtered)} rows")
