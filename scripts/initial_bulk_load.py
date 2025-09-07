"""
Initial Bulk Data Loading Script for BigQuery

This script performs the one-time initial load of 2 years of historical data
for all target symbols into BigQuery. It has been refactored to use modular
components for better maintainability and separation of concerns.

The core loading logic is now in app_modules/bulk_loader.py, while this script
handles command-line interface and orchestration.

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
Refactored: 2025-09-02
"""

import sys
import os
import argparse
from typing import List, Optional

# Add the parent directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_modules.bulk_loader import BulkDataLoader, LoaderConfig


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Bulk load historical stock data into BigQuery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Load all symbols
  %(prog)s --test-mode              # Test mode (5 symbols)
  %(prog)s --resume                 # Resume from checkpoint
  %(prog)s --max-symbols 100        # Limit to 100 symbols
  %(prog)s --symbols AAPL,GOOGL     # Load specific symbols
        """,
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (process only 5 symbols)",
    )

    parser.add_argument(
        "--resume", action="store_true", help="Resume from previous checkpoint"
    )

    parser.add_argument(
        "--max-symbols", type=int, help="Maximum number of symbols to process"
    )

    parser.add_argument(
        "--symbols", type=str, help="Comma-separated list of specific symbols to load"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of symbols to process per batch (default: 10)",
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Rate limit delay between API calls in seconds (default: 0.1)",
    )

    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=50,
        help="Save checkpoint every N symbols (default: 50)",
    )

    return parser.parse_args()


def create_loader_config(args: argparse.Namespace) -> LoaderConfig:
    """
    Create loader configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Configured LoaderConfig instance
    """
    config = LoaderConfig()

    # Apply command line overrides
    config.symbols_per_batch = args.batch_size
    config.rate_limit_delay = args.rate_limit
    config.checkpoint_frequency = args.checkpoint_frequency

    return config


def validate_environment() -> bool:
    """
    Validate that the environment is properly configured.

    Returns:
        True if environment is valid, False otherwise
    """
    try:
        # Check if required modules can be imported
        from app_modules.config import load_environment_variables
        from app_modules.data_handler import load_stock_data
        from app_modules.bigquery_client import upload_to_bigquery

        # Check if configuration is accessible
        alpha_vantage_key, _ = load_environment_variables()
        if not alpha_vantage_key:
            print("Error: Alpha Vantage API key not found in .env file")
            return False

        print("Environment validation passed.")
        return True

    except ImportError as e:
        print(f"Error: Missing required module - {e}")
        return False
    except Exception as e:
        print(f"Error: Environment validation failed - {e}")
        return False


def main() -> None:
    """Main entry point for the bulk loading script."""
    print("Bulk Data Loader for BigQuery")
    print("-" * 40)

    # Parse command line arguments
    args = parse_arguments()

    # Validate environment
    if not validate_environment():
        print("Environment validation failed. Please check your setup.")
        sys.exit(1)

    # Create configuration
    config = create_loader_config(args)

    # Parse symbol list if provided
    symbol_list = None
    if args.symbols:
        symbol_list = [s.strip().upper() for s in args.symbols.split(",")]
        print(f"Processing specific symbols: {symbol_list}")

    # Create and configure loader
    loader = BulkDataLoader(config)

    # Run the bulk loading process
    try:
        results = loader.run_bulk_load(
            symbols=symbol_list,
            test_mode=args.test_mode,
            resume=args.resume,
            max_symbols=args.max_symbols,
        )

        # Print comprehensive summary
        loader.print_final_summary(results)

        # Exit with appropriate code
        if results["failed"] > 0:
            print(f"\nWarning: {results['failed']} symbols failed to load.")
            sys.exit(1)
        else:
            print("\nBulk loading completed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\nBulk loading interrupted by user.")
        sys.exit(2)
    except Exception as e:
        print(f"\nFatal error during bulk loading: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
