"""
Daily Stock Data Updater - Google Cloud Function

This function is designed to run daily via Cloud Scheduler to update stock data
in BigQuery. It only runs on trading days and maintains a rolling window of data.

Key Features:
- Trading day detection using pandas_market_calendars
- Alpha Vantage API integration for stock data
- BigQuery table management with rolling window
- Error handling and logging
- Configurable symbols and date ranges
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

import pandas as pd
import pandas_market_calendars as mcal
from google.cloud import bigquery
import requests
import functions_framework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables (set in Cloud Function configuration)
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
BIGQUERY_PROJECT_ID = os.environ.get("BIGQUERY_PROJECT_ID")
BIGQUERY_DATASET_ID = os.environ.get("BIGQUERY_DATASET_ID", "stock_data")
BIGQUERY_TABLE_ID = os.environ.get("BIGQUERY_TABLE_ID", "raw_stock_data")
MAX_TRADING_DAYS = int(
    os.environ.get("MAX_TRADING_DAYS", "500")
)  # Exactly 500 trading days default

# Default stock symbols to update
DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "SPY", "QQQ", "VTI"]


class TradingDayValidator:
    """Validates if today is a trading day using market calendars."""

    def __init__(self):
        self.nyse = mcal.get_calendar("NYSE")

    def is_trading_day(self, date: datetime = None) -> bool:
        """Check if the given date (or today) is a trading day."""
        if date is None:
            date = datetime.now()

        # Get trading schedule for the date
        schedule = self.nyse.schedule(
            start_date=date.strftime("%Y-%m-%d"), end_date=date.strftime("%Y-%m-%d")
        )

        return len(schedule) > 0

    def get_last_trading_day(self) -> datetime:
        """Get the most recent trading day."""
        today = datetime.now()

        # Look back up to 7 days to find last trading day
        for i in range(7):
            check_date = today - timedelta(days=i)
            if self.is_trading_day(check_date):
                return check_date

        # Fallback: return today if no trading day found
        logger.warning("No trading day found in last 7 days, using today")
        return today


class AlphaVantageClient:
    """Client for fetching stock data from Alpha Vantage API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_daily_data(
        self, symbol: str, outputsize: str = "compact"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch daily stock data from Alpha Vantage.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            outputsize: 'compact' (100 days) or 'full' (20+ years)

        Returns:
            DataFrame with OHLCV data or None if error
        """
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }

        try:
            logger.info(f"Fetching data for {symbol}")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if "Error Message" in data:
                logger.error(
                    f"Alpha Vantage API error for {symbol}: {data['Error Message']}"
                )
                return None

            if "Note" in data:
                logger.warning(f"Alpha Vantage API limit reached: {data['Note']}")
                return None

            # Extract time series data
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                logger.error(f"No time series data found for {symbol}")
                return None

            time_series = data[time_series_key]

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"

            # Rename columns to standard format
            df.columns = ["open", "high", "low", "close", "volume"]

            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            # Add symbol column
            df["symbol"] = symbol

            # Reset index to make date a column
            df = df.reset_index()

            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except requests.RequestException as e:
            logger.error(f"Network error fetching data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {symbol}: {e}")
            return None


class BigQueryManager:
    """Manages BigQuery operations for stock data."""

    def __init__(self, project_id: str, dataset_id: str, table_id: str):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.full_table_id = f"{project_id}.{dataset_id}.{table_id}"

    def ensure_table_exists(self):
        """Create the BigQuery table if it doesn't exist."""
        try:
            self.client.get_table(self.full_table_id)
            logger.info(f"Table {self.full_table_id} already exists")
        except Exception:
            logger.info(f"Creating table {self.full_table_id}")

            schema = [
                bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("open", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("high", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("low", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("close", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("volume", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ]

            table = bigquery.Table(self.full_table_id, schema=schema)
            table = self.client.create_table(table)
            logger.info(f"Created table {self.full_table_id}")

    def upsert_data(self, df: pd.DataFrame):
        """
        Upsert data into BigQuery table.
        Replaces existing records for the same date/symbol combination.
        """
        if df.empty:
            logger.warning("No data to upsert")
            return

        # Add timestamp for when data was updated
        df["updated_at"] = pd.Timestamp.utcnow()

        # Convert date to proper format
        df["date"] = pd.to_datetime(df["date"]).dt.date

        logger.info(f"Upserting {len(df)} records to {self.full_table_id}")

        # Use MERGE to upsert data
        temp_table_id = f"{self.full_table_id}_temp_{int(datetime.now().timestamp())}"

        try:
            # Load data to temporary table
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

            job = self.client.load_table_from_dataframe(
                df, temp_table_id, job_config=job_config
            )
            job.result()  # Wait for job to complete

            # Merge temporary table with main table
            merge_query = f"""
            MERGE `{self.full_table_id}` AS target
            USING `{temp_table_id}` AS source
            ON target.date = source.date AND target.symbol = source.symbol
            WHEN MATCHED THEN
                UPDATE SET
                    open = source.open,
                    high = source.high,
                    low = source.low,
                    close = source.close,
                    volume = source.volume,
                    updated_at = source.updated_at
            WHEN NOT MATCHED THEN
                INSERT (date, symbol, open, high, low, close, volume, updated_at)
                VALUES (source.date, source.symbol, source.open, source.high, 
                       source.low, source.close, source.volume, source.updated_at)
            """

            job = self.client.query(merge_query)
            job.result()

            logger.info(
                f"Successfully upserted data for {df['symbol'].nunique()} symbols"
            )

        finally:
            # Clean up temporary table
            try:
                self.client.delete_table(temp_table_id)
            except Exception as e:
                logger.warning(f"Failed to delete temp table {temp_table_id}: {e}")

    def cleanup_old_data(self, max_trading_days: int):
        """Remove data to maintain exactly max_trading_days per symbol."""

        # Query to keep only the most recent max_trading_days per symbol
        cleanup_query = f"""
        DELETE FROM `{self.full_table_id}`
        WHERE STRUCT(date, symbol) NOT IN (
            SELECT STRUCT(date, symbol)
            FROM (
                SELECT date, symbol,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol 
                           ORDER BY date DESC
                       ) as row_num
                FROM `{self.full_table_id}`
            )
            WHERE row_num <= {max_trading_days}
        )
        """

        job = self.client.query(cleanup_query)
        result = job.result()

        # Get count of rows affected for logging
        rows_deleted = (
            result.num_dml_affected_rows
            if hasattr(result, "num_dml_affected_rows")
            else 0
        )

        logger.info(
            f"Cleaned up {rows_deleted} old records, maintaining {max_trading_days} trading days per symbol"
        )


@functions_framework.http
def daily_stock_update(request):
    """
    Cloud Function entry point for daily stock data updates.

    Can be triggered by:
    1. Cloud Scheduler (recommended for daily updates)
    2. HTTP request with optional parameters

    Request body (JSON, optional):
    {
        "symbols": ["AAPL", "GOOGL"],  // Override default symbols
        "force_update": true,          // Update even if not a trading day
        "outputsize": "full"           // Alpha Vantage outputsize parameter
    }
    """
    try:
        # Parse request body if present
        request_json = {}
        if request.get_json():
            request_json = request.get_json()

        symbols = request_json.get("symbols", DEFAULT_SYMBOLS)
        force_update = request_json.get("force_update", False)
        outputsize = request_json.get("outputsize", "compact")

        logger.info(f"Starting daily stock update for symbols: {symbols}")

        # Validate environment variables
        if not ALPHA_VANTAGE_API_KEY:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable not set")

        # For local testing, allow demo project ID
        bigquery_project_id = BIGQUERY_PROJECT_ID or os.environ.get(
            "BIGQUERY_PROJECT_ID"
        )
        if not bigquery_project_id:
            raise ValueError("BIGQUERY_PROJECT_ID environment variable not set")

        # Check if today is a trading day
        trading_validator = TradingDayValidator()
        if not force_update and not trading_validator.is_trading_day():
            logger.info("Today is not a trading day, skipping update")
            return {
                "status": "skipped",
                "message": "Not a trading day",
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Initialize clients
        av_client = AlphaVantageClient(ALPHA_VANTAGE_API_KEY)
        bq_manager = BigQueryManager(
            bigquery_project_id, BIGQUERY_DATASET_ID, BIGQUERY_TABLE_ID
        )

        # Ensure BigQuery table exists
        bq_manager.ensure_table_exists()

        # Fetch and update data for each symbol
        successful_updates = []
        failed_updates = []

        for symbol in symbols:
            try:
                # Fetch data from Alpha Vantage
                df = av_client.get_daily_data(symbol, outputsize)

                if df is not None and not df.empty:
                    # Upsert to BigQuery
                    bq_manager.upsert_data(df)
                    successful_updates.append(symbol)
                else:
                    failed_updates.append(symbol)

            except Exception as e:
                logger.error(f"Failed to update {symbol}: {e}")
                failed_updates.append(symbol)

        # Clean up old data to maintain exactly 500 trading days per symbol
        try:
            bq_manager.cleanup_old_data(MAX_TRADING_DAYS)
        except Exception as e:
            logger.error(f"Failed to clean up old data: {e}")

        # Prepare response
        response = {
            "status": "completed",
            "successful_updates": successful_updates,
            "failed_updates": failed_updates,
            "total_symbols": len(symbols),
            "success_count": len(successful_updates),
            "failure_count": len(failed_updates),
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Update completed. Success: {len(successful_updates)}, Failed: {len(failed_updates)}"
        )

        return response

    except Exception as e:
        logger.error(f"Critical error in daily_stock_update: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }, 500


if __name__ == "__main__":
    """For local testing."""
    import os
    from unittest.mock import Mock

    # Set environment variables for testing
    os.environ["ALPHA_VANTAGE_API_KEY"] = "demo"
    os.environ["BIGQUERY_PROJECT_ID"] = "your-project-id"

    # Create mock request
    mock_request = Mock()
    mock_request.get_json.return_value = {
        "symbols": ["AAPL"],
        "force_update": True,
        "outputsize": "compact",
    }

    # Test the function
    result = daily_stock_update(mock_request)
    print(json.dumps(result, indent=2))
