from app_modules.bigquery_client import get_bigquery_client
import pandas as pd

try:
    bq = get_bigquery_client()
    result = bq.query_stock_data("GOOG")

    print(f"Total rows: {len(result)}")
    print(f"Unique dates: {len(result.index.unique())}")
    print(f"Duplicate dates: {len(result) - len(result.index.unique())}")

    # Check for actual duplicates
    duplicates = result.index.duplicated()
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate dates:")
        duplicate_dates = result.index[duplicates]
        for date in duplicate_dates.unique():
            print(f"  {date}: {(result.index == date).sum()} occurrences")
            # Show the duplicate rows
            dup_data = result[result.index == date]
            close_values = dup_data["close"].tolist()
            print(f"    Values: {close_values}")
    else:
        print("No duplicate dates found")

    # Show first and last few dates to see the pattern
    print(f"\nFirst 5 dates: {result.index[:5].tolist()}")
    print(f"Last 5 dates: {result.index[-5:].tolist()}")

    # Check if there are weekends in the data
    print(f"\nChecking for weekends:")
    for i, date in enumerate(result.index[:10]):
        weekday = date.strftime("%A")
        print(f"  {date} ({weekday})")

except Exception as e:
    print(f"Error: {e}")
