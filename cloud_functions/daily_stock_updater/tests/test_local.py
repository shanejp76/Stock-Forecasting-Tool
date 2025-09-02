"""
Local testing script for the daily stock updater Cloud Function.

This script allows you to test the Cloud Function locally before deployment.
It simulates the Cloud Functions environment and provides debugging capabilities.
"""

import os
import sys
import json
from datetime import datetime
from unittest.mock import Mock

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main function
from main import daily_stock_update


def setup_test_environment():
    """Set up environment variables for testing."""
    # Set required environment variables
    test_env = {
        "ALPHA_VANTAGE_API_KEY": "demo",  # Use 'demo' for testing without real API key
        "BIGQUERY_PROJECT_ID": "your-test-project-id",
        "BIGQUERY_DATASET_ID": "test_stock_data",
        "BIGQUERY_TABLE_ID": "test_daily_prices",
        "MAX_TRADING_DAYS": "500",  # Exactly 500 trading days for testing
    }

    for key, value in test_env.items():
        os.environ[key] = value

    print("Test environment configured:")
    for key, value in test_env.items():
        print(f"  {key}={value}")
    print()


def create_test_request(test_case="default"):
    """Create mock HTTP request for different test scenarios."""
    mock_request = Mock()

    test_cases = {
        "default": None,  # No request body, use defaults
        "custom_symbols": {
            "symbols": ["AAPL", "GOOGL"],
            "force_update": True,
            "outputsize": "compact",
        },
        "force_update": {"force_update": True},
        "full_data": {"symbols": ["SPY"], "outputsize": "full", "force_update": True},
    }

    request_body = test_cases.get(test_case)
    mock_request.get_json.return_value = request_body

    print(f"Test case: {test_case}")
    if request_body:
        print(f"Request body: {json.dumps(request_body, indent=2)}")
    else:
        print("Request body: None (using defaults)")
    print()

    return mock_request


def run_test(test_case="default"):
    """Run a specific test case."""
    print(f"{'='*60}")
    print(f"Running test: {test_case}")
    print(f"{'='*60}")

    # Set up environment
    setup_test_environment()

    # Create test request
    mock_request = create_test_request(test_case)

    # Run the function
    try:
        print("Executing daily_stock_update function...")
        result = daily_stock_update(mock_request)

        print("\nFunction completed!")
        print("Result:")
        print(json.dumps(result, indent=2))

        return result

    except Exception as e:
        print(f"\nError during execution: {e}")
        return {"status": "error", "error": str(e)}


def main():
    """Main testing function."""
    print("Daily Stock Updater - Local Testing")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Available test cases
    test_cases = ["default", "custom_symbols", "force_update", "full_data"]

    if len(sys.argv) > 1:
        # Run specific test case from command line
        test_case = sys.argv[1]
        if test_case not in test_cases:
            print(f"Invalid test case: {test_case}")
            print(f"Available test cases: {', '.join(test_cases)}")
            sys.exit(1)

        run_test(test_case)
    else:
        # Interactive mode
        print("Available test cases:")
        for i, case in enumerate(test_cases, 1):
            print(f"  {i}. {case}")
        print()

        try:
            choice = input(
                "Enter test case number (or press Enter for default): "
            ).strip()

            if choice == "":
                test_case = "default"
            else:
                test_case = test_cases[int(choice) - 1]

            run_test(test_case)

        except (ValueError, IndexError):
            print("Invalid choice, running default test...")
            run_test("default")
        except KeyboardInterrupt:
            print("\nTest cancelled by user")


if __name__ == "__main__":
    main()
