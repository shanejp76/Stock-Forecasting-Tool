"""
Initial Bulk Data Loading Script for BigQuery

This script performs the one-time initial load of 2 years of historical data
for all target symbols into BigQuery. It processes symbols in batches to
respect API rate limits and provides comprehensive progress tracking.

Enhanced Features:
- Progress tracking with tqdm progress bars
- Checkpoint and resume functionality
- Comprehensive statistics and ETA calculations
- Test mode for validation
- Flexible symbol filtering and limits

Usage:
    python scripts/initial_bulk_load.py [options]

Examples:
    # Load all symbols
    python scripts/initial_bulk_load.py

    # Test mode (5 symbols)
    python scripts/initial_bulk_load.py --test-mode

    # Resume from checkpoint
    python scripts/initial_bulk_load.py --resume

    # Limit to specific number
    python scripts/initial_bulk_load.py --max-symbols 100

    # Specific symbols
    python scripts/initial_bulk_load.py --symbols AAPL,GOOGL,MSFT

Author: Shane
Created: 2025-08-26
Enhanced: 2025-08-26
"""

import sys
import os
import argparse
import time
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set
from tqdm import tqdm

# Add the parent directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_modules.bigquery_client import BigQueryClient
from app_modules.config import load_environment_variables
from app_modules.rate_limiter import AVAPIRateLimiter
from alpha_vantage.timeseries import TimeSeries
from scripts.symbol_universe import (
    get_alpha_vantage_symbols,
    get_filtered_symbol_universe,
)
import pandas as pd

# Import progress tracking
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print(
        "Warning: tqdm not available. Install with 'pip install tqdm' for progress bars."
    )

# Configuration
CHECKPOINT_FILE = "data/bulk_load_progress.pkl"


class BulkDataLoader:
    """Handles initial bulk loading of historical stock data"""

    def __init__(self):
        """Initialize the bulk data loader"""
        self.alpha_vantage_key, _ = load_environment_variables()
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not found in .env file")

        self.ts_av = TimeSeries(key=self.alpha_vantage_key, output_format="pandas")
        self.bq_client = BigQueryClient()

        # Initialize rate limiter for premium tier (75 calls/minute)
        self.rate_limiter = AVAPIRateLimiter(max_calls_per_minute=75)

        # Verify BigQuery connection
        if not self.bq_client.test_connection():
            raise ConnectionError("BigQuery connection failed")

    def get_default_symbols(self) -> List[str]:
        """Get comprehensive universe of symbols from Alpha Vantage"""
        try:
            # Get all symbols from Alpha Vantage
            symbols = get_alpha_vantage_symbols(self.alpha_vantage_key)
            print(f"Retrieved {len(symbols)} symbols from Alpha Vantage")
            return symbols
        except Exception as e:
            print(f"Failed to get symbols from Alpha Vantage: {e}")
            print("Falling back to basic symbol list...")
            # Fallback to basic list
            return [
                "AAPL",
                "GOOGL",
                "MSFT",
                "AMZN",
                "TSLA",
                "NVDA",
                "META",
                "NFLX",
                "AMD",
                "CRM",
                "SPY",
                "QQQ",
                "VTI",
                "IWM",
                "DIA",
            ]

    def load_progress(self) -> Dict[str, Any]:
        """Load progress from checkpoint file."""
        if not os.path.exists(CHECKPOINT_FILE):
            return {
                "completed_symbols": set(),
                "failed_symbols": set(),
                "total_processed": 0,
                "last_checkpoint": None,
            }

        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                progress = pickle.load(f)
            print(f"Loaded checkpoint: {progress['total_processed']} symbols processed")
            return progress
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return {
                "completed_symbols": set(),
                "failed_symbols": set(),
                "total_processed": 0,
                "last_checkpoint": None,
            }

    def save_progress(self, progress: Dict[str, Any]):
        """Save progress to checkpoint file."""
        try:
            os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
            progress["last_checkpoint"] = datetime.now()
            with open(CHECKPOINT_FILE, "wb") as f:
                pickle.dump(progress, f)
            print(f"Progress saved: {progress['total_processed']} symbols processed")
        except Exception as e:
            print(f"Error saving progress: {e}")

    def load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load 2 years of historical data for a single symbol

        Args:
            symbol: Stock symbol to load

        Returns:
            DataFrame with 2 years of data or None if failed
        """
        try:
            print(f"  📊 Fetching data for {symbol}...")

            # Apply rate limiting before making API call
            sleep_time = self.rate_limiter.wait_if_needed()
            if sleep_time > 0:
                print(f"    ⏱️ Rate limited: waited {sleep_time:.1f}s")

            data, meta_data = self.ts_av.get_daily(symbol, outputsize="full")

            if data.empty:
                print(f"  ⚠️ No data received for {symbol}")
                return None

            # Convert to DataFrame with proper column names for free tier API
            data.columns = [
                "open",
                "high", 
                "low",
                "close",
                "volume",
            ]
            
            # Add missing columns with default values for free tier
            data["adjusted_close"] = data["close"]  # Use close as adjusted_close
            data["dividend"] = 0.0
            data["split"] = 1.0
            data.index.name = "date"
            data.reset_index(inplace=True)
            data["date"] = pd.to_datetime(data["date"]).dt.date

            # Filter to last 2 years
            two_years_ago = datetime.now().date() - timedelta(days=730)
            data = data[data["date"] >= two_years_ago].copy()

            print(f"  ✅ Loaded {len(data)} records for {symbol}")
            return data

        except Exception as e:
            print(f"  ❌ Error loading {symbol}: {e}")
            return None

    def load_symbols_batch(self, symbols: List[str], batch_size: int = 5):
        """
        Load data for multiple symbols in batches with progress tracking and checkpointing

        Args:
            symbols: List of symbols to load
            batch_size: Number of symbols to process per batch
        """
        # Load checkpoint data
        progress = self.load_progress()
        completed_symbols = progress["completed_symbols"]
        failed_symbols = progress["failed_symbols"]

        # Filter out already processed symbols
        remaining_symbols = [s for s in symbols if s not in completed_symbols]

        total_symbols = len(symbols)
        total_remaining = len(remaining_symbols)
        successful_loads = len(completed_symbols)
        failed_loads = list(failed_symbols)
        start_time = time.time()

        print(f"🚀 Starting bulk load for {total_symbols} symbols")
        print(f"✅ Already completed: {len(completed_symbols)}")
        print(f"❌ Previously failed: {len(failed_symbols)}")
        print(f"📦 Remaining to process: {total_remaining}")
        print(f"📦 Processing in batches of {batch_size}")

        if total_remaining == 0:
            print("🎉 All symbols already processed!")
            return

        # Show time estimate for remaining symbols
        estimated_time = self.rate_limiter.estimate_time_for_calls(total_remaining)
        print(
            f"⏱️ Estimated time for remaining: {estimated_time:.1f} minutes ({estimated_time/60:.1f} hours)"
        )

        # Show rate limiter status
        self.rate_limiter.log_configuration()
        print("=" * 60)

        # Progress bar for remaining symbols
        with tqdm(total=total_remaining, desc="Loading symbols", unit="symbol") as pbar:
            for i in range(0, total_remaining, batch_size):
                batch = remaining_symbols[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_remaining + batch_size - 1) // batch_size

                pbar.set_description(f"Batch {batch_num}/{total_batches}")

                for symbol in batch:
                    pbar.set_postfix(symbol=symbol)
                    data = self.load_symbol_data(symbol)

                    if data is not None and not data.empty:
                        # Ingest data into BigQuery
                        try:
                            success = self.bq_client.ingest_stock_data(
                                data, symbol, source="alpha_vantage_bulk_load"
                            )
                            if success:
                                successful_loads += 1
                                completed_symbols.add(symbol)
                                pbar.set_postfix(symbol=symbol, status="✅")
                            else:
                                failed_loads.append(symbol)
                                failed_symbols.add(symbol)
                                pbar.set_postfix(symbol=symbol, status="❌ BQ")
                        except Exception as e:
                            print(f"  ❌ BigQuery ingestion failed for {symbol}: {e}")
                            failed_loads.append(symbol)
                            failed_symbols.add(symbol)
                            pbar.set_postfix(symbol=symbol, status="❌ BQ")
                    else:
                        failed_loads.append(symbol)
                        failed_symbols.add(symbol)
                        pbar.set_postfix(symbol=symbol, status="❌ API")

                    # Update progress bar
                    pbar.update(1)

                    # Save checkpoint every 10 symbols
                    if (successful_loads + len(failed_loads)) % 10 == 0:
                        progress_data = {
                            "completed_symbols": completed_symbols,
                            "failed_symbols": failed_symbols,
                            "total_processed": len(completed_symbols)
                            + len(failed_symbols),
                        }
                        self.save_progress(progress_data)

                # Show current rate limiter status
                status = self.rate_limiter.get_status()
                tqdm.write(
                    f"📊 API Usage: {status['current_calls_in_window']}/{status['max_calls_per_minute']} calls in current window"
                )

                # Brief pause between batches for logging/status updates
                if i + batch_size < total_remaining:
                    time.sleep(1)

        # Final checkpoint save
        final_progress = {
            "completed_symbols": completed_symbols,
            "failed_symbols": failed_symbols,
            "total_processed": len(completed_symbols) + len(failed_symbols),
        }
        self.save_progress(final_progress)

        # Summary
        print("\n" + "=" * 60)
        print("📊 BULK LOAD SUMMARY")
        print("=" * 60)
        print(f"✅ Successful loads: {successful_loads}/{total_symbols}")
        print(f"❌ Failed loads: {len(failed_loads)}")
        print(f"📈 Success rate: {(successful_loads/total_symbols)*100:.1f}%")

        if failed_loads:
            print(f"❌ Failed symbols: {', '.join(failed_loads[:10])}")
            if len(failed_loads) > 10:
                print(f"   ... and {len(failed_loads)-10} more")

        actual_time_minutes = (time.time() - start_time) / 60
        print(f"⏱️ Actual time: {actual_time_minutes:.1f} minutes")

        # Performance metrics
        symbols_per_minute = (
            total_remaining / actual_time_minutes if actual_time_minutes > 0 else 0
        )
        print(f"🚀 Throughput: {symbols_per_minute:.1f} symbols/minute")

        print(f"💾 Checkpoint saved to: {CHECKPOINT_FILE}")
        if len(failed_loads) > 0:
            print("⚠️ To retry failed symbols, simply run the script again")


def main():
    """Main function to run bulk data loading"""
    parser = argparse.ArgumentParser(
        description="Initial bulk data loading for BigQuery"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: predefined list)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for processing (default: 5)",
    )
    parser.add_argument(
        "--full-universe",
        action="store_true",
        help="Load all available US stocks from Alpha Vantage",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and proceed automatically",
    )

    args = parser.parse_args()

    try:
        loader = BulkDataLoader()

        # Determine symbols to load
        if args.full_universe:
            print("🌎 Loading full universe of US stocks...")
            from symbol_universe import get_us_stock_universe

            symbols = get_us_stock_universe()
            print(f"📊 Found {len(symbols)} symbols in universe")
        elif args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",")]
        else:
            symbols = loader.get_default_symbols()

        print(f"🎯 Target symbols: {len(symbols)} symbols")
        if len(symbols) <= 20:
            print(f"   {', '.join(symbols)}")

        # Confirm before proceeding
        estimated_time = loader.rate_limiter.estimate_time_for_calls(len(symbols))
        if not args.yes:
            response = input(
                f"\n⚠️ This will load 2 years of data for {len(symbols)} symbols.\n"
                f"⏱️ Estimated time: {estimated_time:.1f} minutes ({estimated_time/60:.1f} hours)\n"
                f"Continue? (y/N): "
            )
            if response.lower() != "y":
                print("❌ Operation cancelled")
                return
        else:
            print(f"\n⚠️ Loading 2 years of data for {len(symbols)} symbols.")
            print(f"⏱️ Estimated time: {estimated_time:.1f} minutes ({estimated_time/60:.1f} hours)")
            print("🚀 Auto-proceeding due to --yes flag")

        # Run bulk load
        loader.load_symbols_batch(symbols, args.batch_size)

        print("\n🎉 Bulk data loading completed!")

    except Exception as e:
        print(f"❌ Bulk loading failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
