"""
Data Ingestion Script for BigQuery

This script demonstrates data ingestion from Alpha Vantage API to BigQuery.
It fetches historical data for a test symbol and loads it into the data warehouse.

Usage:
    python scripts/test_bigquery_ingestion.py

Author: Shane
Created: 2025-08-26
"""

import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_modules.bigquery_client import BigQueryClient
from app_modules.config import load_environment_variables
from alpha_vantage.timeseries import TimeSeries
import pandas as pd


def test_bigquery_ingestion():
    """Test data ingestion from Alpha Vantage to BigQuery"""

    print("🚀 Testing BigQuery Data Ingestion")
    print("=" * 50)

    # Load environment variables
    print("📋 Loading API credentials...")
    alpha_vantage_key, _ = load_environment_variables()

    if not alpha_vantage_key:
        print("❌ Error: Alpha Vantage API key not found in .env file")
        return False

    # Initialize clients
    print("🔌 Initializing BigQuery client...")
    bq_client = BigQueryClient()

    if not bq_client.test_connection():
        print("❌ Error: BigQuery connection failed")
        return False

    print("🔌 Initializing Alpha Vantage client...")
    ts_av = TimeSeries(key=alpha_vantage_key, output_format="pandas")

    # Test symbol
    test_symbol = "AAPL"
    print(f"📈 Fetching data for {test_symbol}...")

    try:
        # Fetch data from Alpha Vantage
        data, meta_data = ts_av.get_daily_adjusted(
            symbol=test_symbol, outputsize="compact"
        )

        # Get the most recent 10 days for testing
        data = data.head(10)

        print(f"📊 Retrieved {len(data)} days of data")
        print("Sample data:")
        print(data.head(3))
        print()

        # Check if symbol already exists in BigQuery
        print("🔍 Checking for existing data in BigQuery...")
        latest_date = bq_client.get_latest_data_date(test_symbol)

        if latest_date:
            print(f"📅 Latest data in BigQuery: {latest_date}")
        else:
            print("📅 No existing data found for this symbol")

        # Ingest data to BigQuery
        print(f"⬆️  Ingesting data to BigQuery...")
        success = bq_client.ingest_stock_data(data, test_symbol, "alpha_vantage_test")

        if success:
            print("✅ Data ingestion successful!")

            # Verify data was inserted
            print("🔍 Verifying data insertion...")
            query_result = bq_client.query_stock_data(test_symbol)

            if not query_result.empty:
                print(
                    f"✅ Verification successful! Found {len(query_result)} rows in BigQuery"
                )
                print("Sample queried data:")
                print(query_result.head(3))

                # Show available symbols
                symbols = bq_client.get_available_symbols()
                print(f"📋 Available symbols in BigQuery: {symbols}")

            else:
                print("⚠️  Warning: No data returned from query")

        else:
            print("❌ Data ingestion failed")
            return False

    except Exception as e:
        print(f"❌ Error during ingestion test: {e}")
        return False

    print("\n🎉 BigQuery ingestion test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_bigquery_ingestion()
    if success:
        print("\n✅ Ready to proceed with Phase 1 implementation!")
    else:
        print("\n❌ Please resolve issues before proceeding.")
