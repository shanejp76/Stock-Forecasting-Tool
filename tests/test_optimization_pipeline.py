#!/usr/bin/env python3
"""
Test Script for Phase 1 Optimization Pipeline

This script tests the complete symbol-specific optimization pipeline:
1. BigQuery optimal parameters table
2. Parameter lookup functionality
3. UI integration
4. Fallback mechanisms

Usage:
    python test_optimization_pipeline.py

Author: Shane
Created: 2025-09-21
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_modules.optimal_parameters import get_optimal_parameters_manager
from app_modules.parameter_lookup import (
    get_optimal_parameters_for_symbol,
    should_use_optimal_parameters,
    get_optimization_status_summary,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_table_creation():
    """Test BigQuery table creation and basic operations."""
    logger.info("Testing BigQuery table creation...")

    try:
        manager = get_optimal_parameters_manager()

        # Test table creation (should succeed or already exist)
        success = manager.create_table()
        if success:
            logger.info("✅ Table creation successful")
        else:
            logger.error("❌ Table creation failed")
            return False

        # Test sample data insertion
        logger.info("Testing sample data insertion...")
        test_success = manager.upsert_parameters(
            symbol="TEST",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            rmse=15.25,
            mae=12.10,
            smape=8.5,
            data_points_used=500,
            cv_folds=5,
        )

        if test_success:
            logger.info("✅ Sample data insertion successful")
        else:
            logger.error("❌ Sample data insertion failed")
            return False

        # Test data retrieval
        logger.info("Testing data retrieval...")
        params = manager.get_parameters("TEST")
        if params:
            logger.info(
                f"✅ Data retrieval successful: {params['symbol']} - "
                f"changepoint={params['changepoint_prior_scale']:.3f}, "
                f"seasonality={params['seasonality_prior_scale']:.1f}"
            )
        else:
            logger.error("❌ Data retrieval failed")
            return False

        # Cleanup test data
        manager.delete_symbol_parameters("TEST")
        logger.info("✅ Test data cleaned up")

        return True

    except Exception as e:
        logger.error(f"❌ Table creation test failed: {e}")
        return False


def test_parameter_lookup():
    """Test parameter lookup functionality."""
    logger.info("Testing parameter lookup functionality...")

    try:
        # Test with non-existent symbol (should return defaults)
        params = get_optimal_parameters_for_symbol("NONEXISTENT")
        if params["source"] in [
            "default_no_optimization",
            "default_fallback",
            "default_error",
        ]:
            logger.info(
                f"✅ Fallback working: {params['source']} for NONEXISTENT symbol"
            )
        else:
            logger.error(f"❌ Unexpected source: {params['source']}")
            return False

        # Test should_use_optimal_parameters
        should_use, reason = should_use_optimal_parameters("AAPL")
        logger.info(
            f"✅ Optimization status check: AAPL - should_use={should_use}, reason={reason}"
        )

        # Test optimization status summary
        status = get_optimization_status_summary()
        logger.info(
            f"✅ Status summary: {status.get('total_optimized', 0)} optimized symbols"
        )

        return True

    except Exception as e:
        logger.error(f"❌ Parameter lookup test failed: {e}")
        return False


def test_optimization_status():
    """Test optimization status and summary functions."""
    logger.info("Testing optimization status functions...")

    try:
        # Test getting symbols with optimization
        manager = get_optimal_parameters_manager()
        symbols = manager.get_all_symbols_with_parameters()
        logger.info(f"✅ Found {len(symbols)} symbols with optimization data")

        if symbols:
            # Test getting summary
            summary = manager.get_optimization_summary()
            if summary is not None and not summary.empty:
                logger.info(f"✅ Summary retrieved for {len(summary)} symbols")
                logger.info(f"  Best RMSE: {summary['rmse'].min():.3f}")
                logger.info(f"  Average RMSE: {summary['rmse'].mean():.3f}")
            else:
                logger.info("ℹ️ No summary data available (empty database)")

        return True

    except Exception as e:
        logger.error(f"❌ Optimization status test failed: {e}")
        return False


def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    logger.info("Testing end-to-end workflow...")

    try:
        # Simulate the main workflow
        test_symbol = "AAPL"

        # 1. Check if optimization exists
        should_use, reason = should_use_optimal_parameters(test_symbol)
        logger.info(
            f"Step 1: Should use optimal for {test_symbol}? {should_use} ({reason})"
        )

        # 2. Get parameters (optimal or default)
        params = get_optimal_parameters_for_symbol(test_symbol)
        logger.info(
            f"Step 2: Got parameters from {params['source']} - "
            f"changepoint={params['changepoint_prior_scale']:.3f}, "
            f"seasonality={params['seasonality_prior_scale']:.1f}"
        )

        # 3. Simulate UI logic
        if params["source"] == "optimized":
            logger.info("Step 3: Would display optimized parameters to user")
        else:
            logger.info(
                "Step 3: Would display default parameters and suggest optimization"
            )

        logger.info("✅ End-to-end workflow test successful")
        return True

    except Exception as e:
        logger.error(f"❌ End-to-end workflow test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🧪 Testing Phase 1: Symbol-Specific Model Optimization Pipeline")
    logger.info("=" * 70)

    tests = [
        ("BigQuery Table Operations", test_table_creation),
        ("Parameter Lookup Functions", test_parameter_lookup),
        ("Optimization Status", test_optimization_status),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 50)

        try:
            success = test_func()
            results[test_name] = success

            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")

        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:<30} {status}")

    logger.info("-" * 70)
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Phase 1 pipeline is ready!")
        return 0
    else:
        logger.error(f"🚨 {total - passed} tests failed. Phase 1 needs fixes.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
