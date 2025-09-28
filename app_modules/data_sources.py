"""
Data Sources Module

This module provides functions for loading stock data from various sources
including BigQuery, Alpha Vantage, and Finnhub APIs.

Functions:
    load_bigquery_data(ticker, start_date): Load data from BigQuery warehouse
    load_alpha_vantage_data(ts_av, ticker): Load data from Alpha Vantage API
    load_finnhub_tickers(api_key, exchange): Load ticker list from Finnhub
    load_stock_data_hybrid(ticker, start_date, use_bigquery, ts_av): Hybrid loading with fallback
    load_bigquery_symbols(): Get available symbols from BigQuery

Author: Shane
Created: 2025-09-02 (Refactored from data_handler.py)
"""

import streamlit as st
import pandas as pd
import requests
from alpha_vantage.timeseries import TimeSeries
from datetime import date, timedelta
from typing import Tuple, List
import os
import json
from app_modules.bigquery_client import get_bigquery_client
from app_modules.deployment_config import deployment_config


def get_bigquery_auth_debug_info() -> str:
    """
    Get detailed authentication debugging information for BigQuery client.
    Returns string with authentication status details.
    """
    debug_parts = []

    # Check Streamlit secrets availability
    try:
        if hasattr(st, "secrets"):
            if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
                debug_parts.append(
                    "FOUND: Streamlit secrets has GOOGLE_APPLICATION_CREDENTIALS_JSON"
                )
                # Try to validate JSON structure
                try:
                    creds_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
                    if isinstance(creds_json, str):
                        creds_data = json.loads(creds_json)
                        project_id = creds_data.get("project_id", "unknown")
                        client_email = creds_data.get("client_email", "unknown")
                        debug_parts.append(f"FOUND: Service account: {client_email}")
                        debug_parts.append(f"FOUND: Project ID: {project_id}")
                    else:
                        debug_parts.append("ERROR: Credentials JSON is not a string")
                except json.JSONDecodeError as e:
                    debug_parts.append(f"ERROR: Invalid JSON in secrets: {str(e)[:50]}")
                except Exception as e:
                    debug_parts.append(f"ERROR: Error parsing secrets: {str(e)[:50]}")
            else:
                debug_parts.append(
                    "MISSING: Streamlit secrets missing GOOGLE_APPLICATION_CREDENTIALS_JSON"
                )
        else:
            debug_parts.append("MISSING: Streamlit secrets not available")
    except Exception as e:
        debug_parts.append(f"ERROR: Error checking Streamlit secrets: {str(e)[:50]}")

    # Check environment variable
    env_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if env_creds:
        debug_parts.append(
            "FOUND: Environment variable GOOGLE_APPLICATION_CREDENTIALS_JSON found"
        )
        try:
            creds_data = json.loads(env_creds)
            project_id = creds_data.get("project_id", "unknown")
            debug_parts.append(f"FOUND: Env project ID: {project_id}")
        except Exception as e:
            debug_parts.append(f"ERROR: Error parsing env credentials: {str(e)[:50]}")
    else:
        debug_parts.append(
            "MISSING: Environment variable GOOGLE_APPLICATION_CREDENTIALS_JSON not found"
        )

    # Check file-based credentials
    gac_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gac_file:
        debug_parts.append(f"FOUND: GOOGLE_APPLICATION_CREDENTIALS file: {gac_file}")
        if os.path.exists(gac_file):
            debug_parts.append("FOUND: Credentials file exists")
        else:
            debug_parts.append("ERROR: Credentials file does not exist")
    else:
        debug_parts.append("MISSING: GOOGLE_APPLICATION_CREDENTIALS file path not set")

    # Check if we're in Streamlit Cloud environment
    if "streamlit.io" in os.getenv("HOSTNAME", ""):
        debug_parts.append("DETECTED: Running on Streamlit Cloud")
    elif os.getenv("STREAMLIT_SHARING_MODE"):
        debug_parts.append("DETECTED: Running in Streamlit sharing mode")
    else:
        debug_parts.append("INFO: Not detected as Streamlit Cloud environment")

    return " | ".join(debug_parts)


@st.cache_data
def load_bigquery_data(ticker: str, start_date: date) -> Tuple[pd.DataFrame, str]:
    """
    Loads daily stock data from BigQuery data warehouse.
    Automatically limits data to exactly 500 most recent trading days to optimize Prophet performance.
    Returns tuple of (data, source) where source indicates data origin.
    """
    # Check if BigQuery is available in this environment
    if not deployment_config.bigquery_available:
        return pd.DataFrame(), "BigQuery disabled in deployment config"

    try:
        # Initialize BigQuery client
        bq_client = get_bigquery_client()

        # Check if client is available and connected
        if not bq_client:
            return pd.DataFrame(), "BigQuery client is None"

        if not bq_client.is_available():
            # Add detailed authentication debugging
            auth_debug_info = get_bigquery_auth_debug_info()
            return (
                pd.DataFrame(),
                f"BigQuery client not available/connected - {auth_debug_info}",
            )  # Query all available data (no date limit initially)
        data = bq_client.query_stock_data(ticker)

        if not data.empty:
            # First reset index to make 'date' a column for deduplication
            data = data.reset_index()

            # Remove duplicates (critical fix for data quality issues)
            original_count = len(data)
            data = data.drop_duplicates(subset=["date"], keep="last")
            duplicate_count = original_count - len(data)

            # Sort by date ascending for Prophet compatibility
            data = data.sort_values("date")

            # Limit to exactly 500 most recent trading days for optimal Prophet performance
            data = data.tail(500)

            # Keep date as column for charting functions that expect data["date"]
            # Don't set it back as index since charts need it as a column
            return data, "BigQuery"
        else:
            return pd.DataFrame(), f"BigQuery returned no data for {ticker}"

    except Exception as e:
        st.warning(f"BigQuery load failed for {ticker}: {str(e)}")
        return pd.DataFrame(), f"BigQuery (error: {str(e)})"


@st.cache_data
def load_stock_data_hybrid(
    ticker: str, start_date: date, use_bigquery: bool = True, _ts_av: TimeSeries = None
) -> Tuple[pd.DataFrame, str]:
    """
    Hybrid data loading with BigQuery primary and Alpha Vantage fallback.
    Returns tuple of (data, source) where source indicates data origin.
    """
    if use_bigquery:
        # Try BigQuery first
        data, source = load_bigquery_data(ticker, start_date)
        if not data.empty:
            return data, source

        # Fall back to Alpha Vantage if BigQuery fails
        st.warning(f"Falling back to Alpha Vantage for {ticker}. Reason: {source}")

        # Add detailed debugging information
        st.info(f"Debug - BigQuery fallback details for {ticker}: {source}")

        # Show deployment config status for additional context
        st.info(
            f"Debug - BigQuery available in deployment config: {deployment_config.bigquery_available}"
        )

    # Load from Alpha Vantage
    if _ts_av is not None:
        data = load_alpha_vantage_data(_ts_av, ticker)
        if not data.empty:
            return data, "Alpha Vantage"

    return pd.DataFrame(), "No data source available"


@st.cache_data
def load_bigquery_symbols() -> List[str]:
    """
    Get list of available stock symbols from BigQuery.
    Returns list of symbol strings.
    """
    # Check if BigQuery is available in this environment
    if not deployment_config.bigquery_available:
        return []

    try:
        bq_client = get_bigquery_client()

        # Check if client is available and connected
        if not bq_client or not bq_client.is_available():
            return []

        symbols = bq_client.get_available_symbols()
        return sorted(symbols)
    except Exception as e:
        st.warning(f"Failed to load BigQuery symbols: {e}")
        return []


@st.cache_data
def load_finnhub_tickers(
    finnhub_api_key: str, exchange_code: str
) -> Tuple[List[str], List[dict]]:
    """
    Loads ticker symbols from Finnhub API.
    Returns tuple of (symbol_list, full_ticker_data).
    """
    url = f"https://finnhub.io/api/v1/stock/symbol?exchange={exchange_code}&token={finnhub_api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        tickers_data = response.json()

        # Extract just the symbols
        tickers = [
            ticker.get("symbol", "") for ticker in tickers_data if ticker.get("symbol")
        ]

        return tickers, tickers_data

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load Finnhub tickers: {e}")
        return [], []
    except Exception as e:
        st.error(f"Error processing Finnhub response: {e}")
        return [], []


@st.cache_data
def load_alpha_vantage_data(_ts_av: TimeSeries, ticker: str) -> pd.DataFrame:
    """
    Loads daily stock data from Alpha Vantage API.
    Returns pandas DataFrame with OHLCV data.
    """
    try:
        data, meta_data = _ts_av.get_daily(symbol=ticker, outputsize="full")

        if data is not None and not data.empty:
            # Rename columns to snake_case format for internal processing
            data.columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
            # Convert index to datetime if it's not already
            data.index = pd.to_datetime(data.index)

            # Reset index to make date available as a column for charting functions
            data = data.reset_index()
            data = data.rename(columns={"index": "date"})

            return data
        else:
            st.warning(f"No data received from Alpha Vantage for {ticker}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading Alpha Vantage data for {ticker}: {e}")
        return pd.DataFrame()
