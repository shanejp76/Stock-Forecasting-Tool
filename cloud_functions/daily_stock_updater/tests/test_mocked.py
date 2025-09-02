"""
Local testing script with mocking for offline testing.

This version mocks BigQuery operations so you can test the function logic
without needing actual GCP credentials or API access.
"""

import os
import sys
import json
from datetime import datetime
from unittest.mock import Mock, patch
import pandas as pd

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_test_environment():
    """Set up environment variables for testing."""
    test_env = {
        "ALPHA_VANTAGE_API_KEY": "demo",
        "BIGQUERY_PROJECT_ID": "test-project-123",
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


def create_mock_stock_data(symbol="AAPL", days=5):
    """Create mock stock data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    data = []
    base_price = 150.0

    for i, date in enumerate(dates):
        price = base_price + (i * 2) + (i % 3)  # Simple price variation
        data.append(
            {
                "date": date,
                "symbol": symbol,
                "open": round(price - 1, 2),
                "high": round(price + 2, 2),
                "low": round(price - 2, 2),
                "close": round(price, 2),
                "volume": 1000000 + (i * 100000),
            }
        )

    return pd.DataFrame(data)


def mock_alpha_vantage_response(symbol):
    """Create a mock Alpha Vantage API response."""
    mock_data = create_mock_stock_data(symbol)

    # Convert to Alpha Vantage format
    time_series = {}
    for _, row in mock_data.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        time_series[date_str] = {
            "1. open": str(row["open"]),
            "2. high": str(row["high"]),
            "3. low": str(row["low"]),
            "4. close": str(row["close"]),
            "5. volume": str(row["volume"]),
        }

    return {"Time Series (Daily)": time_series}


def create_test_request(test_case="default"):
    """Create mock HTTP request for different test scenarios."""
    mock_request = Mock()

    test_cases = {
        "default": None,
        "custom_symbols": {
            "symbols": ["AAPL", "GOOGL"],
            "force_update": True,
            "outputsize": "compact",
        },
        "force_update": {"force_update": True},
        "single_symbol": {"symbols": ["SPY"], "force_update": True},
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


@patch("main.BigQueryManager")
@patch("main.requests.get")
def run_mocked_test(mock_requests_get, mock_bigquery_manager, test_case="default"):
    """Run a test with mocked external dependencies."""

    # Import here to avoid issues with mocking
    from main import daily_stock_update

    print(f"{'='*60}")
    print(f"Running mocked test: {test_case}")
    print(f"{'='*60}")

    # Set up environment
    setup_test_environment()

    # Create test request
    mock_request = create_test_request(test_case)

    # Mock Alpha Vantage API responses
    def mock_api_response(*args, **kwargs):
        # Extract symbol from the request URL params
        params = kwargs.get("params", {})
        symbol = params.get("symbol", "AAPL")

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_alpha_vantage_response(symbol)
        return mock_response

    mock_requests_get.side_effect = mock_api_response

    # Mock BigQuery Manager
    mock_bq_instance = Mock()
    mock_bq_instance.ensure_table_exists.return_value = None
    mock_bq_instance.upsert_data.return_value = None
    mock_bq_instance.cleanup_old_data.return_value = None
    mock_bigquery_manager.return_value = mock_bq_instance

    # Run the function
    try:
        print("Executing daily_stock_update function...")
        result = daily_stock_update(mock_request)

        print("\nFunction completed!")
        print("Result:")
        print(json.dumps(result, indent=2))

        # Show what would have been called
        print("\nMocked operations that would have occurred:")
        print(
            f"- BigQuery table creation: {mock_bq_instance.ensure_table_exists.called}"
        )
        print(f"- Data upserts: {mock_bq_instance.upsert_data.call_count}")
        print(f"- Old data cleanup: {mock_bq_instance.cleanup_old_data.called}")
        print(f"- API calls made: {mock_requests_get.call_count}")

        if mock_bq_instance.upsert_data.called:
            for call in mock_bq_instance.upsert_data.call_args_list:
                df = call[0][0]  # First argument is the DataFrame
                print(
                    f"  - Upserted {len(df)} records for symbols: {df['symbol'].unique().tolist()}"
                )

        return result

    except Exception as e:
        print(f"\nError during execution: {e}")
        return {"status": "error", "error": str(e)}


def main():
    """Main testing function."""
    print("Daily Stock Updater - Local Testing (Mocked)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Available test cases
    test_cases = ["default", "custom_symbols", "force_update", "single_symbol"]

    if len(sys.argv) > 1:
        test_case = sys.argv[1]
        if test_case not in test_cases:
            print(f"Invalid test case: {test_case}")
            print(f"Available test cases: {', '.join(test_cases)}")
            sys.exit(1)

        run_mocked_test(test_case=test_case)
    else:
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

            run_mocked_test(test_case=test_case)

        except (ValueError, IndexError):
            print("Invalid choice, running default test...")
            run_mocked_test(test_case="default")
        except KeyboardInterrupt:
            print("\nTest cancelled by user")


if __name__ == "__main__":
    main()
