from app_modules.data_handler import load_bigquery_data, process_stock_data
from app_modules.bigquery_client import get_bigquery_client
from datetime import date
import pandas as pd


def trace_data_pipeline(ticker="GOOG"):
    """Trace exactly where rows are being filtered in the data pipeline"""

    print(f"=== TRACING DATA PIPELINE FOR {ticker} ===\n")

    # Step 1: Raw BigQuery data
    print("Step 1: Raw BigQuery query")
    bq_client = get_bigquery_client()
    raw_data = bq_client.query_stock_data(ticker)
    print(f"  Raw BigQuery rows: {len(raw_data)}")
    print(f"  Date range: {raw_data.index.min()} to {raw_data.index.max()}")

    # Step 2: Through load_bigquery_data function
    print("\nStep 2: load_bigquery_data function")
    start_date = date(2023, 1, 1)  # Use a date that should include all data
    processed_data, source = load_bigquery_data(ticker, start_date)
    print(f"  Processed rows: {len(processed_data)}")
    if not processed_data.empty:
        print(
            f"  Date range: {processed_data['Date'].min()} to {processed_data['Date'].max()}"
        )

    # Step 3: Through process_stock_data function
    print("\nStep 3: process_stock_data function")
    if not processed_data.empty:
        final_data = process_stock_data(processed_data, start_date)
        print(f"  Final rows: {len(final_data)}")
        if not final_data.empty:
            print(
                f"  Date range: {final_data['Date'].min()} to {final_data['Date'].max()}"
            )

            # Check for any data quality issues
            print(f"\nData Quality Checks:")
            print(f"  Null values: {final_data.isnull().sum().sum()}")
            print(f"  Duplicate dates: {final_data['Date'].duplicated().sum()}")

            # Check date continuity
            final_data_sorted = final_data.sort_values("Date")
            print(
                f"  First 5 dates: {final_data_sorted['Date'].head().dt.date.tolist()}"
            )
            print(
                f"  Last 5 dates: {final_data_sorted['Date'].tail().dt.date.tolist()}"
            )

            # Check for weekend dates (shouldn't be any)
            weekends = final_data_sorted[final_data_sorted["Date"].dt.dayofweek >= 5]
            print(f"  Weekend dates found: {len(weekends)}")
            if len(weekends) > 0:
                print(f"    Weekend dates: {weekends['Date'].dt.date.tolist()}")

    print(f"\n=== SUMMARY ===")
    print(f"Raw BigQuery: {len(raw_data)} rows")
    print(f"After load_bigquery_data: {len(processed_data)} rows")
    if not processed_data.empty:
        print(f"After process_stock_data: {len(final_data)} rows")
        print(f"Total rows lost: {len(raw_data) - len(final_data)}")


if __name__ == "__main__":
    trace_data_pipeline()
