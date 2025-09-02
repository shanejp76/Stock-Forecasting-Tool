#!/usr/bin/env python3
"""
Data Quality Investigation Script
Compares BigQuery vs Alpha Vantage data to identify critical issues:
1. SPY self-correlation showing ~60% instead of 100%
2. AAPL model training failures with BigQuery data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add app_modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "app_modules"))

try:
    from data_handler import (
        load_stock_data_hybrid,
        load_bigquery_data,
        load_alpha_vantage_data,
    )
    from bigquery_client import get_bigquery_client
    from alpha_vantage.timeseries import TimeSeries

    print("✅ Modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and dependencies are installed")
    sys.exit(1)


def compare_data_sources(symbol, days=30):
    """Compare BigQuery vs Alpha Vantage data for a specific symbol"""
    print(f"\n{'='*60}")
    print(f"DATA QUALITY INVESTIGATION: {symbol}")
    print(f"{'='*60}")

    # Setup Alpha Vantage client
    av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not av_api_key:
        print("❌ ALPHA_VANTAGE_API_KEY not found in environment")
        return None, None

    ts_av = TimeSeries(key=av_api_key, output_format="pandas")
    start_date = datetime.now().date() - timedelta(days=days * 2)  # Buffer for weekends

    # Get data from Alpha Vantage
    print(f"\n1. Fetching {symbol} data from Alpha Vantage...")
    try:
        av_data = load_alpha_vantage_data(ts_av, symbol)
        if av_data is not None and not av_data.empty:
            # Convert to proper format for comparison
            av_data = av_data.set_index("Date").tail(days)
            av_data.columns = [col.lower().replace(" ", "_") for col in av_data.columns]
            print(f"   ✅ Alpha Vantage: {len(av_data)} rows")
            print(f"   📅 Date range: {av_data.index.min()} to {av_data.index.max()}")
            print(f"   💰 Latest close: ${av_data['close'].iloc[-1]:.2f}")
        else:
            print(f"   ❌ Alpha Vantage: No data returned")
            av_data = None
    except Exception as e:
        print(f"   ❌ Alpha Vantage error: {e}")
        av_data = None

    print(f"\n2. Fetching {symbol} data from BigQuery...")
    try:
        bq_data, source = load_bigquery_data(symbol, start_date)
        if bq_data is not None and not bq_data.empty:
            bq_data = bq_data.tail(days)  # Last N days
            print(f"   ✅ BigQuery: {len(bq_data)} rows (source: {source})")
            print(f"   📅 Date range: {bq_data.index.min()} to {bq_data.index.max()}")
            print(f"   💰 Latest close: ${bq_data['close'].iloc[-1]:.2f}")
        else:
            print(f"   ❌ BigQuery: No data returned (source: {source})")
            bq_data = None
    except Exception as e:
        print(f"   ❌ BigQuery error: {e}")
        bq_data = None

    if av_data is None or bq_data is None:
        print(f"❌ Cannot compare - missing data from one or both sources")
        return None, None

    # Data structure comparison
    print(f"\n3. Data Structure Comparison:")
    print(f"   Alpha Vantage columns: {list(av_data.columns)}")
    print(f"   BigQuery columns: {list(bq_data.columns)}")

    # Index comparison
    print(f"\n4. Index Comparison:")
    print(f"   Alpha Vantage index type: {type(av_data.index)}")
    print(f"   BigQuery index type: {type(bq_data.index)}")

    # Find common dates
    common_dates = av_data.index.intersection(bq_data.index)
    print(
        f"   Common dates: {len(common_dates)} out of AV:{len(av_data)}, BQ:{len(bq_data)}"
    )

    if len(common_dates) > 0:
        print(f"\n5. Price Comparison (Common Dates):")
        av_common = av_data.loc[common_dates].sort_index()
        bq_common = bq_data.loc[common_dates].sort_index()

        # Compare close prices
        close_diff = (av_common["close"] - bq_common["close"]).abs()
        print(f"   Average close price difference: ${close_diff.mean():.4f}")
        print(f"   Max close price difference: ${close_diff.max():.4f}")
        print(
            f"   Identical close prices: {(close_diff < 0.01).sum()}/{len(close_diff)}"
        )

        # Show sample data
        print(f"\n6. Sample Data Comparison (Last 5 common dates):")
        sample_dates = common_dates[-5:]
        for date in sample_dates:
            av_close = av_common.loc[date, "close"]
            bq_close = bq_common.loc[date, "close"]
            diff = abs(av_close - bq_close)
            print(
                f"   {date}: AV=${av_close:.2f}, BQ=${bq_close:.2f}, Diff=${diff:.4f}"
            )

    return av_data, bq_data


def investigate_spy_correlation():
    """Investigate SPY self-correlation issue"""
    print(f"\n{'='*60}")
    print(f"SPY SELF-CORRELATION INVESTIGATION")
    print(f"{'='*60}")

    av_data, bq_data = compare_data_sources("SPY", days=100)

    if av_data is not None and bq_data is not None:
        # Calculate correlation for common dates
        common_dates = av_data.index.intersection(bq_data.index)
        if len(common_dates) > 10:
            av_closes = av_data.loc[common_dates, "close"].sort_index()
            bq_closes = bq_data.loc[common_dates, "close"].sort_index()

            correlation = av_closes.corr(bq_closes)
            print(f"\n🔍 SPY Correlation Analysis:")
            print(f"   Alpha Vantage vs BigQuery correlation: {correlation:.4f}")
            print(f"   Expected: ~1.0000 (perfect correlation)")
            print(f"   Actual: {correlation:.4f}")

            if correlation < 0.95:
                print(f"   ❌ CRITICAL: Low correlation indicates data quality issues")

                # Check for systematic differences
                price_ratio = (bq_closes / av_closes).describe()
                print(f"\n   Price Ratio Analysis (BQ/AV):")
                print(f"   Mean: {price_ratio['mean']:.6f}")
                print(f"   Std: {price_ratio['std']:.6f}")
                print(f"   Min: {price_ratio['min']:.6f}")
                print(f"   Max: {price_ratio['max']:.6f}")
            else:
                print(f"   ✅ Good correlation - data sources align well")


def investigate_aapl_model_failure():
    """Investigate AAPL model training failure with BigQuery"""
    print(f"\n{'='*60}")
    print(f"AAPL MODEL FAILURE INVESTIGATION")
    print(f"{'='*60}")

    av_data, bq_data = compare_data_sources("AAPL", days=500)

    if bq_data is not None:
        print(f"\n🔍 AAPL BigQuery Data Quality Checks:")

        # Check for missing values
        missing_counts = bq_data.isnull().sum()
        print(f"   Missing values per column:")
        for col, count in missing_counts.items():
            print(f"     {col}: {count}")

        # Check for duplicate dates
        duplicate_dates = bq_data.index.duplicated().sum()
        print(f"   Duplicate dates: {duplicate_dates}")

        # Check for data continuity
        date_gaps = pd.Series(bq_data.index).diff().dt.days
        large_gaps = date_gaps[date_gaps > 7]  # More than a week
        print(f"   Large date gaps (>7 days): {len(large_gaps)}")

        # Check for zero/negative prices
        zero_prices = (bq_data["close"] <= 0).sum()
        print(f"   Zero/negative close prices: {zero_prices}")

        # Check data types
        print(f"   Data types:")
        for col in bq_data.columns:
            print(f"     {col}: {bq_data[col].dtype}")

        # Check for Prophet model requirements
        print(f"\n   Prophet Model Requirements Check:")
        print(f"     Minimum rows needed: 2")
        print(f"     Available rows: {len(bq_data)}")
        print(f"     Date column (ds): {bq_data.index.name or 'index'}")
        print(f"     Value column (y): close")
        print(
            f"     Date range span: {(bq_data.index.max() - bq_data.index.min()).days} days"
        )


def main():
    """Main investigation function"""
    print("STOCK FORECASTING TOOL - DATA QUALITY INVESTIGATION")
    print("=" * 60)
    print("Investigating critical issues:")
    print("1. SPY self-correlation showing ~60% instead of 100%")
    print("2. AAPL model training failures with BigQuery data")

    # Test environment
    print(f"\n📋 Environment Check:")
    try:
        bq_client = get_bigquery_client()
        symbols = bq_client.get_available_symbols()
        print(f"   ✅ BigQuery connection successful")
        print(f"   📊 Available symbols: {len(symbols)}")
        print(f"   🔤 Sample symbols: {symbols[:5]}")
    except Exception as e:
        print(f"   ❌ BigQuery connection failed: {e}")
        return

    # Run investigations
    investigate_spy_correlation()
    investigate_aapl_model_failure()

    print(f"\n{'='*60}")
    print("INVESTIGATION COMPLETE")
    print("Review the output above to identify data quality issues")
    print("='*60")


if __name__ == "__main__":
    main()
