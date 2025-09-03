"""
Bulk Data Loader Core Module

This module provides the core BulkDataLoader class and related functionality
for loading historical stock data into BigQuery with progress tracking,
checkpointing, and comprehensive error handling.

Classes:
    BulkDataLoader: Main class for managing bulk data loading operations
    LoaderConfig: Configuration class for loader settings
    LoaderStats: Statistics tracking class

Author: Shane
Created: 2025-09-02 (Refactored from initial_bulk_load.py)
"""

import os
import sys
import time
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set
from tqdm import tqdm

# Add the parent directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_modules.config import Config
from app_modules.data_handler import load_stock_data
from app_modules.bigquery_client import upload_to_bigquery


class LoaderConfig:
    """Configuration settings for the bulk data loader."""

    def __init__(self):
        self.symbols_per_batch: int = 10
        self.max_retries: int = 3
        self.retry_delay: int = 5
        self.checkpoint_frequency: int = 50
        self.rate_limit_delay: float = 0.1
        self.test_symbol_limit: int = 5
        self.progress_file: str = "data/bulk_load_progress.pkl"

        # Data collection settings
        self.years_of_history: int = 2
        self.end_date: datetime = datetime.now()
        self.start_date: datetime = self.end_date - timedelta(
            days=365 * self.years_of_history
        )


class LoaderStats:
    """Statistics tracking for bulk data loading operations."""

    def __init__(self):
        self.total_symbols: int = 0
        self.processed_symbols: int = 0
        self.successful_loads: int = 0
        self.failed_loads: int = 0
        self.skipped_symbols: int = 0
        self.start_time: datetime = datetime.now()
        self.last_checkpoint_time: datetime = datetime.now()
        self.errors: List[Dict[str, Any]] = []

    def add_error(self, symbol: str, error: str, error_type: str = "general") -> None:
        """Add an error to the tracking list."""
        self.errors.append(
            {
                "symbol": symbol,
                "error": error,
                "error_type": error_type,
                "timestamp": datetime.now(),
            }
        )

    def get_eta(self) -> str:
        """Calculate estimated time to completion."""
        if self.processed_symbols == 0:
            return "Unknown"

        elapsed = datetime.now() - self.start_time
        rate = self.processed_symbols / elapsed.total_seconds()
        remaining = self.total_symbols - self.processed_symbols

        if rate > 0:
            eta_seconds = remaining / rate
            eta = timedelta(seconds=int(eta_seconds))
            return str(eta)
        return "Unknown"

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of loading statistics."""
        elapsed = datetime.now() - self.start_time
        return {
            "total_symbols": self.total_symbols,
            "processed": self.processed_symbols,
            "successful": self.successful_loads,
            "failed": self.failed_loads,
            "skipped": self.skipped_symbols,
            "elapsed_time": str(elapsed),
            "eta": self.get_eta(),
            "success_rate": (self.successful_loads / max(1, self.processed_symbols))
            * 100,
            "error_count": len(self.errors),
        }


class BulkDataLoader:
    """
    Manages the bulk loading of historical stock data into BigQuery.

    Features:
    - Progress tracking with tqdm
    - Checkpoint and resume functionality
    - Comprehensive error handling and retry logic
    - Rate limiting and batch processing
    - Detailed statistics and logging
    """

    def __init__(self, config: Optional[LoaderConfig] = None):
        """
        Initialize the bulk data loader.

        Args:
            config: Optional configuration object. Uses defaults if not provided.
        """
        self.config = config or LoaderConfig()
        self.stats = LoaderStats()
        self.processed_symbols: Set[str] = set()
        self.failed_symbols: Set[str] = set()

        # Initialize app configuration
        self.app_config = Config()

        # Progress tracking
        self.progress_bar: Optional[tqdm] = None

    def load_checkpoint(self) -> bool:
        """
        Load progress from checkpoint file.

        Returns:
            True if checkpoint was loaded successfully, False otherwise.
        """
        try:
            if os.path.exists(self.config.progress_file):
                with open(self.config.progress_file, "rb") as f:
                    checkpoint = pickle.load(f)

                self.processed_symbols = checkpoint.get("processed_symbols", set())
                self.failed_symbols = checkpoint.get("failed_symbols", set())
                self.stats.successful_loads = checkpoint.get("successful_loads", 0)
                self.stats.failed_loads = checkpoint.get("failed_loads", 0)
                self.stats.processed_symbols = len(self.processed_symbols)

                print(
                    f"Checkpoint loaded: {len(self.processed_symbols)} symbols already processed"
                )
                return True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")

        return False

    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        try:
            os.makedirs(os.path.dirname(self.config.progress_file), exist_ok=True)

            checkpoint = {
                "processed_symbols": self.processed_symbols,
                "failed_symbols": self.failed_symbols,
                "successful_loads": self.stats.successful_loads,
                "failed_loads": self.stats.failed_loads,
                "timestamp": datetime.now(),
            }

            with open(self.config.progress_file, "wb") as f:
                pickle.dump(checkpoint, f)

        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def get_target_symbols(self) -> List[str]:
        """
        Get the list of target symbols for bulk loading.

        Returns:
            List of stock symbols to process.
        """
        # This should be replaced with actual symbol discovery logic
        # For now, using a sample list
        symbols = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "AMD",
            "NFLX",
            "DIS",
            "BABA",
            "V",
            "MA",
            "JPM",
            "JNJ",
            "PG",
            "KO",
            "PEP",
            "WMT",
            "HD",
            "VZ",
            "T",
            "CVX",
            "XOM",
            "BA",
            "CAT",
            "MMM",
            "GE",
            "IBM",
            "INTC",
            "CSCO",
            "ORCL",
            "CRM",
            "ADBE",
            "PYPL",
            "SQ",
            "ROKU",
            "SPOT",
            "UBER",
            "LYFT",
            "SNAP",
            "TWTR",
            "PINS",
            "ZM",
            "PTON",
            "DOCU",
        ]

        return symbols

    def load_symbol_data(self, symbol: str) -> bool:
        """
        Load data for a single symbol with retry logic.

        Args:
            symbol: Stock symbol to load data for

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(self.config.max_retries):
            try:
                # Load stock data
                data = load_stock_data(
                    symbol,
                    self.config.start_date.strftime("%Y-%m-%d"),
                    self.config.end_date.strftime("%Y-%m-%d"),
                )

                if data.empty:
                    self.stats.add_error(symbol, "No data returned", "no_data")
                    return False

                # Upload to BigQuery
                success = upload_to_bigquery(data, symbol)

                if success:
                    return True
                else:
                    self.stats.add_error(
                        symbol, "BigQuery upload failed", "upload_error"
                    )

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}: {str(e)}"
                self.stats.add_error(symbol, error_msg, "exception")

                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        return False

    def process_symbols_batch(self, symbols: List[str]) -> None:
        """
        Process a batch of symbols.

        Args:
            symbols: List of symbols to process in this batch
        """
        for symbol in symbols:
            if symbol in self.processed_symbols:
                self.stats.skipped_symbols += 1
                if self.progress_bar:
                    self.progress_bar.update(1)
                continue

            # Rate limiting
            time.sleep(self.config.rate_limit_delay)

            # Attempt to load data
            success = self.load_symbol_data(symbol)

            # Update tracking
            self.processed_symbols.add(symbol)
            self.stats.processed_symbols += 1

            if success:
                self.stats.successful_loads += 1
            else:
                self.stats.failed_loads += 1
                self.failed_symbols.add(symbol)

            # Update progress bar
            if self.progress_bar:
                self.progress_bar.set_postfix(
                    {
                        "Success": self.stats.successful_loads,
                        "Failed": self.stats.failed_loads,
                        "ETA": self.stats.get_eta(),
                    }
                )
                self.progress_bar.update(1)

    def run_bulk_load(
        self,
        symbols: Optional[List[str]] = None,
        test_mode: bool = False,
        resume: bool = False,
        max_symbols: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the bulk data loading process.

        Args:
            symbols: Optional list of specific symbols to load
            test_mode: If True, only process a small number of symbols
            resume: If True, attempt to resume from checkpoint
            max_symbols: Maximum number of symbols to process

        Returns:
            Dictionary with loading results and statistics
        """
        print("Starting bulk data loading process...")

        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()

        # Get target symbols
        if symbols:
            target_symbols = symbols
        else:
            target_symbols = self.get_target_symbols()

        # Apply limits
        if test_mode:
            target_symbols = target_symbols[: self.config.test_symbol_limit]
        elif max_symbols:
            target_symbols = target_symbols[:max_symbols]

        # Filter out already processed symbols if resuming
        if resume:
            target_symbols = [
                s for s in target_symbols if s not in self.processed_symbols
            ]

        self.stats.total_symbols = len(target_symbols)

        print(f"Processing {len(target_symbols)} symbols...")
        print(
            f"Date range: {self.config.start_date.date()} to {self.config.end_date.date()}"
        )

        # Initialize progress bar
        self.progress_bar = tqdm(
            total=len(target_symbols), desc="Loading symbols", unit="symbol"
        )

        try:
            # Process symbols in batches
            for i in range(0, len(target_symbols), self.config.symbols_per_batch):
                batch = target_symbols[i : i + self.config.symbols_per_batch]
                self.process_symbols_batch(batch)

                # Save checkpoint periodically
                if (i + len(batch)) % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving checkpoint...")
            self.save_checkpoint()

        finally:
            if self.progress_bar:
                self.progress_bar.close()

        # Final checkpoint save
        self.save_checkpoint()

        # Return results
        return self.get_final_results()

    def get_final_results(self) -> Dict[str, Any]:
        """
        Get the final results and statistics.

        Returns:
            Dictionary with comprehensive results
        """
        results = self.stats.get_summary()
        results.update(
            {
                "failed_symbols": list(self.failed_symbols),
                "recent_errors": self.stats.errors[-10:] if self.stats.errors else [],
                "checkpoint_file": self.config.progress_file,
            }
        )

        return results

    def print_final_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a comprehensive summary of the loading process.

        Args:
            results: Results dictionary from get_final_results()
        """
        print("\n" + "=" * 60)
        print("BULK DATA LOADING SUMMARY")
        print("=" * 60)
        print(f"Total symbols processed: {results['processed']}")
        print(f"Successful loads: {results['successful']}")
        print(f"Failed loads: {results['failed']}")
        print(f"Skipped symbols: {results['skipped']}")
        print(f"Success rate: {results['success_rate']:.1f}%")
        print(f"Total elapsed time: {results['elapsed_time']}")

        if results["failed_symbols"]:
            print(f"\nFailed symbols ({len(results['failed_symbols'])}):")
            for symbol in results["failed_symbols"][:10]:  # Show first 10
                print(f"  - {symbol}")
            if len(results["failed_symbols"]) > 10:
                print(f"  ... and {len(results['failed_symbols']) - 10} more")

        if results["recent_errors"]:
            print(f"\nRecent errors:")
            for error in results["recent_errors"][-3:]:  # Show last 3
                print(f"  - {error['symbol']}: {error['error']}")

        print(f"\nCheckpoint saved to: {results['checkpoint_file']}")
        print("=" * 60)
