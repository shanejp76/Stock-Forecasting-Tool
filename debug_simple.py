#!/usr/bin/env python3
"""
Quick debug script to show the column name issue
"""
import pandas as pd

# Simulate what happens in the data pipeline
print("=== COLUMN NAME DEBUGGING ===\n")

# 1. Simulate raw data from BigQuery (has snake_case columns)
raw_data = pd.DataFrame(
    {
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "close": [100, 101, 102],
        "high": [105, 106, 107],
        "low": [95, 96, 97],
        "RSI": [30, 40, 50],
    }
)

print("1. RAW DATA (from BigQuery):")
print(f"   Columns: {list(raw_data.columns)}")
print(f"   Has 'date' column: {'date' in raw_data.columns}")
print()

# 2. Simulate the prepare_data_for_display function
display_data = raw_data.copy()

# Mapping from snake_case to Title Case for display
column_mapping = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adjusted_close": "Adjusted Close",
    "volume": "Volume",
}

# Apply mapping only for columns that exist
for snake_case, proper_case in column_mapping.items():
    if snake_case in display_data.columns:
        display_data = display_data.rename(columns={snake_case: proper_case})

print("2. DISPLAY DATA (after prepare_data_for_display):")
print(f"   Columns: {list(display_data.columns)}")
print(f"   Has 'date' column: {'date' in display_data.columns}")
print(f"   Has 'Date' column: {'Date' in display_data.columns}")
print()

# 3. Show what indicator_charts.py tries to do
print("3. WHAT INDICATOR_CHARTS.PY EXPECTS:")
print("   Code: x=data['date']")
print("   But display_data has: 'Date' (Title Case)")
print("   Result: KeyError: 'date'")
print()

print("4. THE PROBLEM:")
print("   • prepare_data_for_display() converts 'date' → 'Date'")
print("   • indicator_charts.py still expects 'date' (lowercase)")
print("   • This creates a mismatch and KeyError")
print()

print("5. SOLUTION OPTIONS:")
print("   A. Change indicator_charts.py to use 'Date' (Title Case)")
print("   B. Don't convert date column in prepare_data_for_display()")
print("   C. Update prepare_data_for_display() to handle this properly")
