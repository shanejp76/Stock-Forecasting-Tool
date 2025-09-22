#!/usr/bin/env python3
"""
Symbol-Specific Prophet Optimization Script

This script performs hyperparameter optimization for Prophet models on each symbol
in the BigQuery warehouse. It extracts the methodology from archive notebooks
(2024.12.18 - Model Eval.ipynb) and stores optimal parameters in BigQuery.

The script uses grid search optimization with cross-validation to find the best
changepoint_prior_scale and seasonality_prior_scale parameters for each symbol.

Usage:
    python scripts/optimize_symbol_parameters.py [options]

Examples:
    # Optimize all symbols
    python scripts/optimize_symbol_parameters.py

    # Optimize specific symbols
    python scripts/optimize_symbol_parameters.py --symbols AAPL,GOOGL,MSFT

    # Test mode with limited parameter grid
    python scripts/optimize_symbol_parameters.py --test-mode

    # Force re-optimization of existing symbols
    python scripts/optimize_symbol_parameters.py --force-reoptimize

Author: Shane
Created: 2025-09-21
Based on: Archive/2024.12.18 - Model Eval.ipynb methodology
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_modules.bigquery_client import get_bigquery_client
from app_modules.optimal_parameters import get_optimal_parameters_manager
from app_modules.data_sources import load_bigquery_data

# Suppress Prophet warnings for cleaner output
warnings.filterwarnings("ignore", message=".*Importing plotly failed.*")
warnings.filterwarnings(
    "ignore", message=".*The parameter 'n_changepoints' is ignored.*"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f'optimization_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
    ],
)
logger = logging.getLogger(__name__)


class SymbolOptimizer:
    """Optimizes Prophet parameters for individual stock symbols."""

    def __init__(self, test_mode: bool = False):
        """
        Initialize the optimizer.

        Args:
            test_mode: If True, use reduced parameter grid for faster testing
        """
        self.test_mode = test_mode
        self.bq_client = get_bigquery_client()
        self.params_manager = get_optimal_parameters_manager()

        # Parameter grids based on archive notebook methodology
        if test_mode:
            # Reduced grid for testing
            self.param_grid = {
                "changepoint_prior_scale": [0.01, 0.1],
                "seasonality_prior_scale": [0.1, 1],
            }
            self.cv_horizon = "15 days"  # Shorter for testing
            self.cv_period = "7 days"
        else:
            # Full grid from archive notebook
            self.param_grid = {
                "changepoint_prior_scale": [0.001, 0.01, 0.1, 1],
                "seasonality_prior_scale": [0.01, 0.1, 1, 10],
            }
            self.cv_horizon = "30 days"  # Standard cross-validation
            self.cv_period = "15 days"

    def prepare_data_for_prophet(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load and prepare data for Prophet training.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame prepared for Prophet or None on error
        """
        try:
            logger.info(f"Loading data for {symbol}")

            # Load data from BigQuery (gets last 500 trading days)
            data, source = load_bigquery_data(symbol, datetime.now().date())

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # Prepare for Prophet (needs 'ds' and 'y' columns)
            prophet_data = (
                data.reset_index() if "date" not in data.columns else data.copy()
            )

            # Ensure we have the right column names
            if "date" in prophet_data.columns:
                prophet_data = prophet_data.rename(columns={"date": "ds"})
            elif prophet_data.index.name == "date":
                prophet_data = prophet_data.reset_index().rename(columns={"date": "ds"})

            # Use 'close' price as target variable
            if "close" not in prophet_data.columns:
                logger.error(f"No 'close' column found for {symbol}")
                return None

            prophet_data = prophet_data[["ds", "close"]].rename(columns={"close": "y"})

            # Ensure datetime format
            prophet_data["ds"] = pd.to_datetime(prophet_data["ds"])

            # Sort by date
            prophet_data = prophet_data.sort_values("ds").reset_index(drop=True)

            logger.info(
                f"Prepared {len(prophet_data)} data points for {symbol} "
                f"from {prophet_data['ds'].min().date()} to {prophet_data['ds'].max().date()}"
            )

            return prophet_data

        except Exception as e:
            logger.error(f"Failed to prepare data for {symbol}: {e}")
            return None

    def optimize_symbol(
        self, symbol: str, force_reoptimize: bool = False
    ) -> Optional[Dict]:
        """
        Optimize Prophet parameters for a single symbol using grid search.

        Args:
            symbol: Stock symbol
            force_reoptimize: If True, skip existing parameter check

        Returns:
            Dict with optimization results or None on failure
        """
        try:
            # Check if parameters already exist (unless forcing)
            if not force_reoptimize:
                existing_params = self.params_manager.get_parameters(symbol)
                if existing_params:
                    logger.info(
                        f"Optimal parameters already exist for {symbol}, skipping"
                    )
                    return existing_params

            logger.info(f"Starting optimization for {symbol}")

            # Prepare data
            prophet_data = self.prepare_data_for_prophet(symbol)
            if prophet_data is None:
                return None

            # Ensure minimum data points for cross-validation
            if len(prophet_data) < 60:  # Need at least 2 months of data
                logger.warning(
                    f"Insufficient data for {symbol}: {len(prophet_data)} points"
                )
                return None

            # Grid search optimization (from archive notebook methodology)
            best_params = None
            best_metrics = None
            best_rmse = float("inf")

            total_combinations = len(self.param_grid["changepoint_prior_scale"]) * len(
                self.param_grid["seasonality_prior_scale"]
            )
            current_combination = 0

            logger.info(
                f"Testing {total_combinations} parameter combinations for {symbol}"
            )

            for changepoint_prior_scale in self.param_grid["changepoint_prior_scale"]:
                for seasonality_prior_scale in self.param_grid[
                    "seasonality_prior_scale"
                ]:
                    current_combination += 1

                    try:
                        logger.debug(
                            f"Testing {symbol} [{current_combination}/{total_combinations}]: "
                            f"changepoint={changepoint_prior_scale}, seasonality={seasonality_prior_scale}"
                        )

                        # Create and fit Prophet model
                        model = Prophet(
                            changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_prior_scale=seasonality_prior_scale,
                            daily_seasonality=False,  # Disable for stock data
                            weekly_seasonality=True,
                            yearly_seasonality=True,
                        )

                        # Fit model
                        model.fit(prophet_data)

                        # Perform cross-validation
                        cv_results = cross_validation(
                            model,
                            horizon=self.cv_horizon,
                            period=self.cv_period,
                            parallel="processes",
                        )

                        if cv_results.empty:
                            logger.warning(
                                f"Cross-validation returned no results for {symbol}"
                            )
                            continue

                        # Calculate performance metrics
                        metrics = performance_metrics(cv_results, rolling_window=1)

                        if metrics.empty:
                            logger.warning(
                                f"Performance metrics calculation failed for {symbol}"
                            )
                            continue

                        # Get average metrics
                        avg_rmse = metrics["rmse"].mean()
                        avg_mae = metrics["mae"].mean()
                        avg_smape = metrics["smape"].mean()

                        logger.debug(
                            f"  RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, SMAPE: {avg_smape:.4f}"
                        )

                        # Update best parameters if this is better
                        if avg_rmse < best_rmse:
                            best_rmse = avg_rmse
                            best_params = {
                                "changepoint_prior_scale": changepoint_prior_scale,
                                "seasonality_prior_scale": seasonality_prior_scale,
                            }
                            best_metrics = {
                                "rmse": avg_rmse,
                                "mae": avg_mae,
                                "smape": avg_smape,
                            }

                            logger.info(
                                f"New best for {symbol}: RMSE={avg_rmse:.4f} "
                                f"(changepoint={changepoint_prior_scale}, seasonality={seasonality_prior_scale})"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Parameter combination failed for {symbol}: {e}"
                        )
                        continue

            if best_params is None:
                logger.error(f"No valid parameter combinations found for {symbol}")
                return None

            # Store results in BigQuery
            logger.info(f"Storing optimal parameters for {symbol}")

            success = self.params_manager.upsert_parameters(
                symbol=symbol,
                changepoint_prior_scale=best_params["changepoint_prior_scale"],
                seasonality_prior_scale=best_params["seasonality_prior_scale"],
                rmse=best_metrics["rmse"],
                mae=best_metrics["mae"],
                smape=best_metrics["smape"],
                data_points_used=len(prophet_data),
                cv_folds=len(
                    cross_validation(
                        model, horizon=self.cv_horizon, period=self.cv_period
                    )
                ),
            )

            if success:
                logger.info(f"SUCCESS: Optimization complete for {symbol}")
                logger.info(
                    f"  Best parameters: changepoint_prior_scale={best_params['changepoint_prior_scale']}, "
                    f"seasonality_prior_scale={best_params['seasonality_prior_scale']}"
                )
                logger.info(
                    f"  Performance: RMSE={best_metrics['rmse']:.4f}, "
                    f"MAE={best_metrics['mae']:.4f}, SMAPE={best_metrics['smape']:.4f}"
                )

                return {
                    "symbol": symbol,
                    "parameters": best_params,
                    "metrics": best_metrics,
                    "data_points": len(prophet_data),
                }
            else:
                logger.error(f"Failed to store parameters for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Optimization failed for {symbol}: {e}")
            return None

    def optimize_multiple_symbols(
        self, symbols: List[str], force_reoptimize: bool = False
    ) -> Dict[str, Optional[Dict]]:
        """
        Optimize parameters for multiple symbols.

        Args:
            symbols: List of stock symbols
            force_reoptimize: If True, re-optimize existing symbols

        Returns:
            Dict mapping symbol to optimization results
        """
        results = {}

        logger.info(f"Starting optimization for {len(symbols)} symbols")
        logger.info(f"Test mode: {self.test_mode}")
        logger.info(f"Force re-optimization: {force_reoptimize}")

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing symbol {i}/{len(symbols)}: {symbol}")
            logger.info(f"{'='*60}")

            result = self.optimize_symbol(symbol, force_reoptimize)
            results[symbol] = result

            if result:
                logger.info(f"✅ Success for {symbol}")
            else:
                logger.error(f"❌ Failed for {symbol}")

        return results

    def print_summary(self, results: Dict[str, Optional[Dict]]):
        """Print optimization summary."""
        successful = [
            symbol for symbol, result in results.items() if result is not None
        ]
        failed = [symbol for symbol, result in results.items() if result is None]

        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total symbols processed: {len(results)}")
        logger.info(f"Successful optimizations: {len(successful)}")
        logger.info(f"Failed optimizations: {len(failed)}")

        if successful:
            logger.info(f"\n✅ Successful symbols: {', '.join(successful)}")

        if failed:
            logger.info(f"\n❌ Failed symbols: {', '.join(failed)}")

        # Show best performing symbols
        if successful:
            logger.info(f"\n📊 Top 5 performing symbols by RMSE:")
            perf_data = []
            for symbol in successful:
                result = results[symbol]
                perf_data.append(
                    {
                        "symbol": symbol,
                        "rmse": result["metrics"]["rmse"],
                        "mae": result["metrics"]["mae"],
                        "smape": result["metrics"]["smape"],
                    }
                )

            perf_df = pd.DataFrame(perf_data).sort_values("rmse")
            for _, row in perf_df.head().iterrows():
                logger.info(
                    f"  {row['symbol']:>6}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, SMAPE={row['smape']:.4f}"
                )


def main():
    """Main entry point for the optimization script."""
    parser = argparse.ArgumentParser(
        description="Optimize Prophet parameters for stock symbols"
    )

    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols to optimize (e.g., AAPL,GOOGL,MSFT)",
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use reduced parameter grid for faster testing",
    )

    parser.add_argument(
        "--force-reoptimize",
        action="store_true",
        help="Force re-optimization of symbols that already have parameters",
    )

    parser.add_argument(
        "--max-symbols", type=int, help="Maximum number of symbols to process"
    )

    args = parser.parse_args()

    logger.info("Symbol-Specific Prophet Optimization")
    logger.info("=" * 50)

    # Initialize optimizer
    optimizer = SymbolOptimizer(test_mode=args.test_mode)

    # Determine symbols to process
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        logger.info(f"Processing specified symbols: {symbols}")
    else:
        # Get all symbols from BigQuery
        logger.info("Getting all symbols from BigQuery...")
        symbols = optimizer.bq_client.get_available_symbols()
        if not symbols:
            logger.error("No symbols found in BigQuery")
            sys.exit(1)

        logger.info(f"Found {len(symbols)} symbols in BigQuery")

        # Apply max symbols limit
        if args.max_symbols and len(symbols) > args.max_symbols:
            symbols = symbols[: args.max_symbols]
            logger.info(f"Limited to first {args.max_symbols} symbols")

    # Run optimization
    start_time = datetime.now()
    results = optimizer.optimize_multiple_symbols(symbols, args.force_reoptimize)
    end_time = datetime.now()

    # Print summary
    optimizer.print_summary(results)

    duration = end_time - start_time
    logger.info(f"\nTotal execution time: {duration}")
    logger.info(f"Average time per symbol: {duration / len(symbols) if symbols else 0}")

    # Exit code based on results
    successful_count = sum(1 for result in results.values() if result is not None)
    if successful_count == 0:
        logger.error("No symbols were successfully optimized")
        sys.exit(1)
    elif successful_count < len(results):
        logger.warning(
            f"Only {successful_count}/{len(results)} symbols were successfully optimized"
        )
        sys.exit(2)
    else:
        logger.info("All symbols were successfully optimized!")
        sys.exit(0)


if __name__ == "__main__":
    main()
