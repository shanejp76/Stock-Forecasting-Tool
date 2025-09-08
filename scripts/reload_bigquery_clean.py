#!/usr/bin/env python3
"""
Clean BigQuery Data Reload Script

This script clears the existing BigQuery data and reloads a small set of symbols
with fresh unadjusted data using our corrected data pipeline.

Usage:
    python scripts/reload_bigquery_clean.py
"""

import sys
import os
from datetime import datetime

# Add the parent directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_modules.bigquery_client import get_bigquery_client
from app_modules.config import load_environment_variables
from alpha_vantage.timeseries import TimeSeries
import pandas as pd


def clear_bigquery_table():
    """Clear all data from the BigQuery table"""
    print("🗑️  Clearing BigQuery table...")

    client = get_bigquery_client()

    # Delete all records from the table
    delete_query = """
    DELETE FROM `stock-forecasting-tool-2025.stock_data.raw_stock_data`
    WHERE TRUE
    """

    try:
        job = client.client.query(delete_query)
        result = job.result()
        print(f"✅ Cleared BigQuery table successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to clear BigQuery table: {e}")
        return False


def load_symbol_data(symbol: str, ts_av: TimeSeries, bq_client) -> bool:
    """Load fresh unadjusted data for a single symbol"""
    print(f"📈 Loading data for {symbol}...")

    try:
        # Fetch unadjusted data from Alpha Vantage
        data, meta_data = ts_av.get_daily(symbol=symbol, outputsize="compact")

        if data.empty:
            print(f"  ⚠️ No data received for {symbol}")
            return False

        # Rename columns to match BigQuery schema
        data.columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        # Convert index to datetime and reset for BigQuery
        data.index = pd.to_datetime(data.index)
        data = data.reset_index()
        data = data.rename(columns={"index": "date"})

        # Take only the most recent 10 days for testing
        data = data.head(10)

        # Upload to BigQuery
        success = bq_client.ingest_stock_data(data, symbol, "alpha_vantage_clean")

        if success:
            print(f"  ✅ Successfully loaded {len(data)} rows for {symbol}")
            return True
        else:
            print(f"  ❌ Failed to upload data for {symbol}")
            return False

    except Exception as e:
        print(f"  ❌ Error loading {symbol}: {e}")
        return False


def main():
    """Main function to clear and reload BigQuery data"""
    print("🚀 BigQuery Clean Data Reload")
    print("=" * 50)

    # Load environment variables
    print("📋 Loading API credentials...")
    alpha_vantage_key, _ = load_environment_variables()

    if not alpha_vantage_key:
        print("❌ Error: Alpha Vantage API key not found in .env file")
        return False

    # Initialize clients
    print("🔌 Initializing clients...")
    bq_client = get_bigquery_client()
    ts_av = TimeSeries(key=alpha_vantage_key, output_format="pandas")

    if not bq_client.test_connection():
        print("❌ Error: BigQuery connection failed")
        return False

    # Test symbols (small set for validation)
    test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    print(f"🎯 Target symbols: {test_symbols}")

    # Ask for confirmation
    response = input(
        "\n⚠️  This will DELETE ALL current BigQuery data and reload with fresh unadjusted data. Continue? (y/N): "
    )
    if response.lower() != "y":
        print("❌ Operation cancelled")
        return False

    # Clear existing data
    if not clear_bigquery_table():
        print("❌ Failed to clear table. Aborting.")
        return False

    # Load fresh data
    print(f"\n📊 Loading fresh unadjusted data for {len(test_symbols)} symbols...")
    successful = 0
    failed = 0

    for symbol in test_symbols:
        if load_symbol_data(symbol, ts_av, bq_client):
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 50)
    print("📊 RELOAD SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully loaded: {successful} symbols")
    print(f"❌ Failed to load: {failed} symbols")
    print(f"📈 Total symbols processed: {successful + failed}")

    if successful > 0:
        print(f"\n🎉 BigQuery reload completed successfully!")
        print(f"💡 Ready to test with clean unadjusted data")
        return True
    else:
        print(f"\n❌ All symbol loads failed. Please check errors above.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
