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
        self._connection_error_details = None
        self._auth_method_used = None
        self._last_test_results = {}

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
                        self._auth_method_used = "streamlit_secrets"
                except Exception as e:
                    logger.debug(f"Failed to access Streamlit secrets: {e}")
                    self._connection_error_details = {
                        "auth_step": "streamlit_secrets_access",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }

            # Priority 2: Try environment variable (for other deployed environments)
            if not service_account_key:
                service_account_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                if service_account_key:
                    logger.info(
                        "Using service account credentials from environment variable"
                    )
                    self._auth_method_used = "environment_variable"

            if service_account_key:
                # Parse service account key from JSON string
                try:
                    credentials_info = json.loads(service_account_key)
                    self.credentials = (
                        service_account.Credentials.from_service_account_info(
                            credentials_info
                        )
                    )
                    self.project = credentials_info.get("project_id", PROJECT_ID)
                except json.JSONDecodeError as e:
                    self._connection_error_details = {
                        "auth_step": "json_parsing",
                        "error": f"Invalid JSON in service account key: {str(e)}",
                        "error_type": "JSONDecodeError",
                        "auth_method": self._auth_method_used,
                    }
                    logger.error(f"JSON decode error for service account: {e}")
                    self._connection_available = False
                    return
                except Exception as e:
                    self._connection_error_details = {
                        "auth_step": "credentials_creation",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "auth_method": self._auth_method_used,
                    }
                    logger.error(
                        f"Failed to create credentials from service account: {e}"
                    )
                    self._connection_available = False
                    return
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
                    self._auth_method_used = "default_authentication"
                except Exception as auth_error:
                    logger.warning(
                        f"Default authentication failed (expected on deployed environments): {auth_error}"
                    )
                    self._connection_error_details = {
                        "auth_step": "default_authentication",
                        "error": str(auth_error),
                        "error_type": type(auth_error).__name__,
                    }
                    # Don't raise exception, just mark as unavailable
                    self._connection_available = False
                    return

            try:
                self.client = bigquery.Client(
                    credentials=self.credentials, project=PROJECT_ID
                )
                self.dataset_ref = self.client.dataset(DATASET_ID)
            except Exception as e:
                self._connection_error_details = {
                    "auth_step": "bigquery_client_creation",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "auth_method": self._auth_method_used,
                    "project_id": PROJECT_ID,
                }
                logger.error(f"Failed to create BigQuery client: {e}")
                self._connection_available = False
                return

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
            self._connection_error_details = {
                "auth_step": "initialization_general",
                "error": str(e),
                "error_type": type(e).__name__,
                "auth_method": self._auth_method_used,
            }
            self._connection_available = False

    def is_available(self) -> bool:
        """Check if BigQuery connection is available"""
        return self._connection_available

    def get_detailed_status(self) -> dict:
        """
        Return comprehensive connection status for debugging

        Returns:
            dict: Detailed status information including auth method, test results, and error details
        """
        status = {
            "connection_available": self._connection_available,
            "auth_method_used": self._auth_method_used,
            "project_id": self.project,
            "dataset_id": DATASET_ID,
            "raw_table_id": RAW_TABLE_ID,
            "streamlit_available": STREAMLIT_AVAILABLE,
            "client_exists": self.client is not None,
            "credentials_exists": self.credentials is not None,
            "connection_error_details": self._connection_error_details,
            "last_test_results": self._last_test_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add environment context
        if STREAMLIT_AVAILABLE and hasattr(st, "secrets"):
            try:
                status["streamlit_secrets_available"] = (
                    "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets
                )
            except:
                status["streamlit_secrets_available"] = "unable_to_check"
        else:
            status["streamlit_secrets_available"] = False

        status["env_var_available"] = bool(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        )

        return status

    def test_connection(self) -> bool:
        """Test BigQuery connection with detailed step-by-step diagnostics"""
        self._last_test_results = {
            "step_1_client_available": False,
            "step_2_simple_query": False,
            "step_3_project_access": False,
            "step_4_dataset_access": False,
            "step_5_table_access": False,
            "error_details": {},
            "auth_method_used": self._auth_method_used,
            "project_id": self.project,
            "dataset_id": DATASET_ID,
        }

        # Step 1: Check if client exists
        if not self.client:
            self._last_test_results["error_details"][
                "step_1"
            ] = "BigQuery client is None"
            logger.error("BigQuery client is None - cannot test connection")
            return False

        self._last_test_results["step_1_client_available"] = True
        logger.info("Step 1 PASSED: BigQuery client object exists")

        # Step 2: Test simple query execution
        try:
            query = f"SELECT 1 as test_connection LIMIT 1"
            job_config = bigquery.QueryJobConfig()
            job_config.use_query_cache = True
            job_config.maximum_bytes_billed = 1024

            query_job = self.client.query(query, job_config=job_config, timeout=10)
            results = query_job.result(timeout=10)
            list(results)  # Consume the result

            self._last_test_results["step_2_simple_query"] = True
            logger.info("Step 2 PASSED: Simple query execution successful")
        except Exception as e:
            self._last_test_results["error_details"]["step_2"] = {
                "error": str(e),
                "error_type": type(e).__name__,
            }
            logger.error(f"Step 2 FAILED: Simple query execution failed: {e}")
            return False

        # Step 3: Test project access
        try:
            # Use a simple query to verify project access instead of get_project (not available)
            query = f"SELECT '{self.project}' as project_id"
            job_config = bigquery.QueryJobConfig()
            job_config.use_query_cache = True
            job_config.maximum_bytes_billed = 1024

            query_job = self.client.query(query, job_config=job_config, timeout=10)
            results = query_job.result(timeout=10)
            list(results)  # Consume the result

            self._last_test_results["step_3_project_access"] = True
            self._last_test_results["project_verified"] = self.project
            logger.info(f"Step 3 PASSED: Project access successful - {self.project}")
        except Exception as e:
            self._last_test_results["error_details"]["step_3"] = {
                "error": str(e),
                "error_type": type(e).__name__,
                "project_id_attempted": self.project,
            }
            logger.error(
                f"Step 3 FAILED: Project access failed for {self.project}: {e}"
            )
            return False

        # Step 4: Test dataset access
        try:
            dataset = self.client.get_dataset(f"{self.project}.{DATASET_ID}")
            self._last_test_results["step_4_dataset_access"] = True
            self._last_test_results["dataset_location"] = dataset.location
            logger.info(
                f"Step 4 PASSED: Dataset access successful - location: {dataset.location}"
            )
        except Exception as e:
            self._last_test_results["error_details"]["step_4"] = {
                "error": str(e),
                "error_type": type(e).__name__,
                "dataset_id_attempted": f"{self.project}.{DATASET_ID}",
            }
            logger.error(f"Step 4 FAILED: Dataset access failed for {DATASET_ID}: {e}")
            return False

        # Step 5: Test table access
        try:
            table = self.client.get_table(f"{self.project}.{DATASET_ID}.{RAW_TABLE_ID}")
            self._last_test_results["step_5_table_access"] = True
            self._last_test_results["table_num_rows"] = table.num_rows
            self._last_test_results["table_schema_fields"] = len(table.schema)
            logger.info(
                f"Step 5 PASSED: Table access successful - {table.num_rows} rows, {len(table.schema)} fields"
            )
        except Exception as e:
            self._last_test_results["error_details"]["step_5"] = {
                "error": str(e),
                "error_type": type(e).__name__,
                "table_id_attempted": f"{self.project}.{DATASET_ID}.{RAW_TABLE_ID}",
            }
            logger.error(f"Step 5 FAILED: Table access failed for {RAW_TABLE_ID}: {e}")
            return False

        logger.info("All connection test steps PASSED: BigQuery fully operational")
        return True

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
