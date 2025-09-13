"""
BigQuery Client Module for Stock Forecasting Tool

This module provides a client for interacting with Google BigQuery for stock data
storage and retrieval. It handles data ingestion, querying, and basic operations
for the data warehouse layer.

Functions:
    get_bigquery_client(): Initialize and return BigQuery client
    ingest_stock_data(data, symbol): Insert stock data into BigQuery
    query_stock_data(symbol, start_date, end_date): Retrieve stock data from BigQuery
    get_latest_data_date(symbol): Get the most recent data date for a symbol

Author: Shane
Created: 2025-08-26
"""

import pandas as pd
from google.cloud import bigquery
from google.auth import default
from google.oauth2 import service_account
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import logging
import os
import json

# Import streamlit for secrets access (optional dependency)
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery configuration from environment
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "stock-forecasting-tool-2025")
DATASET_ID = "stock_data"
RAW_TABLE_ID = "raw_stock_data"
INDICATORS_TABLE_ID = "technical_indicators"


class BigQueryClient:
    """Client for interacting with BigQuery stock data warehouse"""

    def __init__(self):
        """Initialize BigQuery client with authentication"""
        self.client = None
        self.credentials = None
        self.project = PROJECT_ID
        self.dataset_ref = None
        self._connection_available = False

        try:
            # Priority 1: Try Streamlit secrets (for Streamlit Cloud deployment)
            service_account_key = None

            if STREAMLIT_AVAILABLE and hasattr(st, "secrets"):
                try:
                    service_account_key = st.secrets.get(
                        "GOOGLE_APPLICATION_CREDENTIALS_JSON", None
                    )
                    if service_account_key:
                        logger.info(
                            "Using service account credentials from Streamlit secrets"
                        )
                except Exception as e:
                    logger.debug(f"Failed to access Streamlit secrets: {e}")

            # Priority 2: Try environment variable (for other deployed environments)
            if not service_account_key:
                service_account_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                if service_account_key:
                    logger.info(
                        "Using service account credentials from environment variable"
                    )

            if service_account_key:
                # Parse service account key from JSON string
                credentials_info = json.loads(service_account_key)
                self.credentials = (
                    service_account.Credentials.from_service_account_info(
                        credentials_info
                    )
                )
                self.project = credentials_info.get("project_id", PROJECT_ID)
            else:
                # Priority 3: Fallback to default authentication (for local development only)
                # Add timeout to prevent hanging on Streamlit Cloud
                try:
                    import google.auth.transport.requests
                    import google.auth._default

                    # Set a short timeout for metadata service
                    request = google.auth.transport.requests.Request(timeout=5)
                    self.credentials, self.project = default(request=request)
                    logger.info("Using default authentication with timeout")
                except Exception as auth_error:
                    logger.warning(
                        f"Default authentication failed (expected on deployed environments): {auth_error}"
                    )
                    # Don't raise exception, just mark as unavailable
                    self._connection_available = False
                    return

            self.client = bigquery.Client(
                credentials=self.credentials, project=PROJECT_ID
            )
            self.dataset_ref = self.client.dataset(DATASET_ID)

            # Test connection
            self._connection_available = self.test_connection()

            if self._connection_available:
                logger.info(
                    f"BigQuery client initialized successfully for project: {PROJECT_ID}"
                )
            else:
                logger.warning("BigQuery client initialized but connection test failed")

        except Exception as e:
            logger.warning(f"BigQuery client initialization failed: {e}")
            logger.info("Application will run with Alpha Vantage data source only")
            self._connection_available = False

    def is_available(self) -> bool:
        """Check if BigQuery connection is available"""
        return self._connection_available

    def test_connection(self) -> bool:
        """Test BigQuery connection with timeout handling"""
        if not self.client:
            return False

        try:
            # Use a lightweight query with timeout for connection testing
            query = f"SELECT 1 as test_connection LIMIT 1"

            # Configure job with timeout
            job_config = bigquery.QueryJobConfig()
            job_config.use_query_cache = True
            job_config.maximum_bytes_billed = 1024  # Minimal billing

            # Execute query with timeout
            query_job = self.client.query(query, job_config=job_config, timeout=10)
            results = query_job.result(timeout=10)  # 10 second timeout
            list(results)  # Consume the result

            logger.info("BigQuery connection test successful")
            return True
        except Exception as e:
            logger.warning(f"BigQuery connection test failed: {e}")
            return False

    def ingest_stock_data(
        self, data: pd.DataFrame, symbol: str, source: str = "alpha_vantage"
    ) -> bool:
        """
        Insert stock data into BigQuery

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            source: Data source identifier

        Returns:
            bool: Success status
        """
        try:
            # Prepare data for BigQuery
            data_copy = data.copy()
            data_copy["symbol"] = symbol
            data_copy["ingested_at"] = datetime.utcnow()
            data_copy["source"] = source

            # Ensure date column is properly formatted
            if "Date" in data_copy.columns:
                data_copy["date"] = pd.to_datetime(data_copy["Date"]).dt.date
                data_copy.drop("Date", axis=1, inplace=True)
            elif data_copy.index.name == "Date" or "date" in str(data_copy.index.dtype):
                data_copy["date"] = pd.to_datetime(data_copy.index).date
                data_copy.reset_index(drop=True, inplace=True)

            # Map column names to BigQuery schema (OHLCV only)
            column_mapping = {
                # Alpha Vantage daily (unadjusted) column names
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
                # Alternative column names (for different endpoints)
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }

            for old_col, new_col in column_mapping.items():
                if old_col in data_copy.columns:
                    data_copy[new_col] = data_copy[old_col]
                    data_copy.drop(old_col, axis=1, inplace=True)

            # Fill missing values for OHLCV columns
            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            for col in numeric_columns:
                if col in data_copy.columns:
                    data_copy[col] = pd.to_numeric(
                        data_copy[col], errors="coerce"
                    ).fillna(0)

            # Insert data into BigQuery
            table_ref = self.dataset_ref.table(RAW_TABLE_ID)
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                autodetect=False,
            )

            job = self.client.load_table_from_dataframe(
                data_copy, table_ref, job_config=job_config
            )
            job.result()  # Wait for job to complete

            logger.info(f"Successfully ingested {len(data_copy)} rows for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to ingest data for {symbol}: {e}")
            return False

    def query_stock_data(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """
        Query stock data from BigQuery

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)

        Returns:
            DataFrame with stock data
        """
        try:
            base_query = f"""
            SELECT 
                date,
                open,
                high,
                low,
                close,
                volume
            FROM `{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE_ID}`
            WHERE symbol = @symbol
            """

            # Add date filters if provided
            if start_date:
                base_query += " AND date >= @start_date"
            if end_date:
                base_query += " AND date <= @end_date"

            base_query += " ORDER BY date ASC"

            # Configure query parameters
            query_parameters = [
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
            ]

            if start_date:
                query_parameters.append(
                    bigquery.ScalarQueryParameter("start_date", "DATE", start_date)
                )
            if end_date:
                query_parameters.append(
                    bigquery.ScalarQueryParameter("end_date", "DATE", end_date)
                )

            job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)

            # Execute query
            result = self.client.query(base_query, job_config=job_config).to_dataframe()

            if not result.empty:
                result.set_index("date", inplace=True)
                logger.info(f"Retrieved {len(result)} rows for {symbol}")
            else:
                logger.warning(f"No data found for {symbol}")

            return result

        except Exception as e:
            logger.error(f"Failed to query data for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_data_date(self, symbol: str) -> Optional[str]:
        """
        Get the most recent data date for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Latest date as string (YYYY-MM-DD) or None
        """
        try:
            query = f"""
            SELECT MAX(date) as latest_date
            FROM `{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE_ID}`
            WHERE symbol = @symbol
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                ]
            )

            result = self.client.query(query, job_config=job_config).result()

            for row in result:
                if row.latest_date:
                    return row.latest_date.strftime("%Y-%m-%d")

            return None

        except Exception as e:
            logger.error(f"Failed to get latest date for {symbol}: {e}")
            return None

    def get_available_symbols(self) -> List[str]:
        """
        Get list of all symbols with data in BigQuery

        Returns:
            List of stock symbols
        """
        try:
            query = f"""
            SELECT DISTINCT symbol
            FROM `{PROJECT_ID}.{DATASET_ID}.{RAW_TABLE_ID}`
            ORDER BY symbol
            """

            result = self.client.query(query).result()
            symbols = [row.symbol for row in result]

            logger.info(f"Found {len(symbols)} symbols in BigQuery")
            return symbols

        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []


# Convenience function to get a client instance
def get_bigquery_client() -> BigQueryClient:
    """Get BigQuery client instance"""
    return BigQueryClient()


# Convenience function for bulk loading operations
def upload_to_bigquery(
    data: pd.DataFrame, symbol: str, source: str = "alpha_vantage"
) -> bool:
    """
    Upload stock data to BigQuery using the BigQuery client.

    Args:
        data: DataFrame with stock data
        symbol: Stock symbol
        source: Data source identifier

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        client = get_bigquery_client()
        return client.ingest_stock_data(data, symbol, source)
    except Exception as e:
        logger.error(f"Failed to upload data for {symbol}: {e}")
        return False
