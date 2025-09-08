"""
Clear BigQuery Data Script

This script clears all data from the BigQuery daily_prices table to prepare for
a fresh reload with updated, consistent unadjusted data.

Usage:
    python scripts/clear_bigquery_data.py [--confirm]

Author: GitHub Copilot
Created: 2025-09-06
"""

import sys
import os
import argparse

# Add the parent directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_modules.bigquery_client import get_bigquery_client
from google.cloud import bigquery


def clear_bigquery_data(confirm: bool = False) -> bool:
    """
    Clear all data from the BigQuery daily_prices table.

    Args:
        confirm: If True, skip confirmation prompt

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("🔥 BigQuery Data Clearing Tool")
        print("=" * 50)

        # Initialize BigQuery client
        bq_client = get_bigquery_client()

        # Get current data count
        symbols = bq_client.get_available_symbols()
        print(f"📊 Current data: {len(symbols)} symbols in BigQuery")

        if not confirm:
            print("\n⚠️  WARNING: This will delete ALL data from BigQuery!")
            print("This action cannot be undone.")
            response = input("\nType 'DELETE' to confirm: ")

            if response != "DELETE":
                print("❌ Operation cancelled")
                return False

        # Clear the table
        print("\n🗑️  Clearing BigQuery table...")

        # Use the internal client to execute delete query
        delete_query = """
        DELETE FROM `stock-forecasting-tool-2025.stock_data.raw_stock_data`
        WHERE TRUE
        """

        job = bq_client.client.query(delete_query)
        result = job.result()

        print("✅ BigQuery data cleared successfully!")

        # Verify clearing
        symbols_after = bq_client.get_available_symbols()
        print(f"📊 Symbols remaining: {len(symbols_after)}")

        return True

    except Exception as e:
        print(f"❌ Error clearing BigQuery data: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Clear BigQuery data")
    parser.add_argument(
        "--confirm", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    success = clear_bigquery_data(confirm=args.confirm)

    if success:
        print("\n✅ Ready to reload with updated data!")
        print("Next step: Run bulk loader with fixed functions")
    else:
        print("\n❌ Failed to clear data")
        sys.exit(1)


if __name__ == "__main__":
    main()
