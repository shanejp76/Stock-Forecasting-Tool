#!/usr/bin/env python3
"""
Debug script to check column names in the forecast data.
This will help us understand what's happening with the 'date' column issue.
"""

import pandas as pd
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app_modules.data_handler import load_data
from app_modules.config import load_config
from app_modules.chart_layout import prepare_data_for_display


def debug_data_columns():
    """Debug the column names at each step of the data pipeline."""

    print("=== DEBUGGING COLUMN NAMES ===")

    # Load configuration
    config = load_config()
    print(f"Config loaded: {type(config)}")

    # Load data for AAPL (a known symbol)
    test_symbol = "AAPL"
    print(f"\n1. Loading data for {test_symbol}...")

    try:
        data = load_data(test_symbol, use_bigquery=True)
        print(f"   Raw data shape: {data.shape}")
        print(f"   Raw data columns: {list(data.columns)}")
        print(f"   Raw data dtypes:\n{data.dtypes}")

        if not data.empty:
            print(f"   First few rows of raw data:")
            print(data.head())

            # Check if 'date' column exists
            has_date = "date" in data.columns
            print(f"\n   Has 'date' column: {has_date}")

            if not has_date:
                # Check for index
                print(f"   Index name: {data.index.name}")
                print(f"   Index type: {type(data.index)}")
                print(f"   First few index values: {data.index[:5].tolist()}")

            # Test prepare_data_for_display function
            print(f"\n2. Testing prepare_data_for_display function...")
            display_data = prepare_data_for_display(data)
            print(f"   Display data shape: {display_data.shape}")
            print(f"   Display data columns: {list(display_data.columns)}")

            has_display_date = "Date" in display_data.columns
            print(f"   Has 'Date' column (Title Case): {has_display_date}")

            # Check what the indicator_charts.py expects vs gets
            print(f"\n3. Analyzing what indicator_charts.py expects...")
            print(f"   indicator_charts.py expects: data['date'] (lowercase)")
            print(f"   But display_data has: {list(display_data.columns)}")

            if "Date" in display_data.columns:
                print(
                    f"   ISSUE: display_data has 'Date' (Title Case) but code expects 'date' (lowercase)"
                )

        else:
            print("   ERROR: Data is empty!")

    except Exception as e:
        print(f"   ERROR loading data: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_data_columns()
