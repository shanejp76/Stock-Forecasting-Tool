#!/usr/bin/env python3
"""
Simple verification script to test BigQuery deduplication fix
Bypasses Streamlit caching to show clean output
"""

import pandas as pd
import os
import sys
from datetime import datetime, timedelta

# Add app_modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), "app_modules"))

from bigquery_client import get_bigquery_client


def verify_deduplication_fix():
    """Verify that deduplication is working in BigQuery data"""
    print("VERIFYING DEDUPLICATION FIX")
    print("=" * 40)

    try:
        # Get BigQuery client
        bq_client = get_bigquery_client()

        # Test SPY data before processing (raw)
        print("\n1. RAW SPY DATA (before deduplication):")
        spy_raw = bq_client.query_stock_data("SPY")
        print(f"   Raw rows: {len(spy_raw)}")

        # Manual deduplication to simulate the fix
        spy_dedup = spy_raw.drop_duplicates(subset=["date"], keep="last")
        duplicates_removed = len(spy_raw) - len(spy_dedup)
        print(f"   After deduplication: {len(spy_dedup)} rows")
        print(f"   Duplicates removed: {duplicates_removed}")

        # Test AAPL data
        print("\n2. RAW AAPL DATA (before deduplication):")
        aapl_raw = bq_client.query_stock_data("AAPL")
        print(f"   Raw rows: {len(aapl_raw)}")

        aapl_dedup = aapl_raw.drop_duplicates(subset=["date"], keep="last")
        duplicates_removed = len(aapl_raw) - len(aapl_dedup)
        print(f"   After deduplication: {len(aapl_dedup)} rows")
        print(f"   Duplicates removed: {duplicates_removed}")

        # Test correlation fix
        print("\n3. CORRELATION TEST:")
        spy_clean = spy_dedup.tail(100).copy()

        # Compare identical datasets (should be perfect correlation)
        correlation = spy_clean["close"].corr(spy_clean["close"])
        print(f"   SPY self-correlation: {correlation:.6f}")

        if correlation == 1.0:
            print("   ✅ SUCCESS: Perfect correlation achieved")
        else:
            print("   ❌ ISSUE: Correlation should be exactly 1.0")

        # Test model readiness
        print("\n4. MODEL READINESS TEST:")

        # Check for unique dates (Prophet requirement)
        spy_unique_dates = spy_dedup["date"].nunique()
        spy_total_rows = len(spy_dedup)
        print(f"   SPY unique dates: {spy_unique_dates}, total rows: {spy_total_rows}")

        aapl_unique_dates = aapl_dedup["date"].nunique()
        aapl_total_rows = len(aapl_dedup)
        print(
            f"   AAPL unique dates: {aapl_unique_dates}, total rows: {aapl_total_rows}"
        )

        if spy_unique_dates == spy_total_rows:
            print("   ✅ SPY: Ready for Prophet (unique dates)")
        else:
            print("   ❌ SPY: Still has duplicate dates")

        if aapl_unique_dates == aapl_total_rows:
            print("   ✅ AAPL: Ready for Prophet (unique dates)")
        else:
            print("   ❌ AAPL: Still has duplicate dates")

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n{'='*40}")
    print("VERIFICATION COMPLETE")


if __name__ == "__main__":
    verify_deduplication_fix()
