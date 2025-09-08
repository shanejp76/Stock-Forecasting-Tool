#!/usr/bin/env python3
"""
Detailed BigQuery Data Structure Investigation
"""

import pandas as pd
import numpy as np
import os
import sys

# Add app_modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "app_modules"))

from bigquery_client import get_bigquery_client


def investigate_bigquery_structure():
    """Investigate the actual BigQuery data structure"""
    print("BIGQUERY DATA STRUCTURE INVESTIGATION")
    print("=" * 50)

    try:
        # Get BigQuery client
        bq_client = get_bigquery_client()

        # Get raw data for SPY
        print("\n1. RAW SPY DATA FROM BIGQUERY:")
        spy_data = bq_client.query_stock_data("SPY")

        print(f"   Data type: {type(spy_data)}")
        print(f"   Shape: {spy_data.shape}")
        print(f"   Columns: {list(spy_data.columns)}")
        print(f"   Index type: {type(spy_data.index)}")
        print(f"   Index name: {spy_data.index.name}")

        # Show first few rows
        print(f"\n   First 5 rows:")
        print(spy_data.head())

        print(f"\n   Last 5 rows:")
        print(spy_data.tail())

        print(f"\n   Data types:")
        print(spy_data.dtypes)

        # Check for null values
        print(f"\n   Null values:")
        print(spy_data.isnull().sum())

        # Check date range
        if hasattr(spy_data.index, "min"):
            print(f"\n   Date range: {spy_data.index.min()} to {spy_data.index.max()}")

        # Get raw data for AAPL
        print("\n\n2. RAW AAPL DATA FROM BIGQUERY:")
        aapl_data = bq_client.query_stock_data("AAPL")

        print(f"   Data type: {type(aapl_data)}")
        print(f"   Shape: {aapl_data.shape}")
        print(f"   Columns: {list(aapl_data.columns)}")
        print(f"   Index type: {type(aapl_data.index)}")

        # Show sample data
        print(f"\n   First 5 rows:")
        print(aapl_data.head())

        print(f"\n   Last 5 rows:")
        print(aapl_data.tail())

        # Compare with expected structure
        print(f"\n\n3. EXPECTED VS ACTUAL STRUCTURE:")
        print(f"   Expected columns: ['open', 'high', 'low', 'close', 'volume']")
        print(f"   SPY actual columns: {list(spy_data.columns)}")
        print(f"   AAPL actual columns: {list(aapl_data.columns)}")

        # Test direct BigQuery query
        print(f"\n\n4. DIRECT BIGQUERY QUERY TEST:")
        query = """
        SELECT symbol, date, open, high, low, close, volume
        FROM `stock-forecasting-tool-2025.stock_data.raw_stock_data`
        WHERE symbol = 'SPY'
        ORDER BY date DESC
        LIMIT 5
        """

        result = bq_client.client.query(query).to_dataframe()
        print(f"   Direct query result shape: {result.shape}")
        print(f"   Direct query columns: {list(result.columns)}")
        print(f"   Direct query sample:")
        print(result)

    except Exception as e:
        print(f"❌ Investigation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    investigate_bigquery_structure()
