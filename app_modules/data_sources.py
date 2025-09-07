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
from app_modules.bigquery_client import get_bigquery_client


@st.cache_data
def load_bigquery_data(ticker: str, start_date: date) -> Tuple[pd.DataFrame, str]:
    """
    Loads daily stock data from BigQuery data warehouse.
    Automatically limits data to exactly 500 most recent trading days to optimize Prophet performance.
    Returns tuple of (data, source) where source indicates data origin.
    """
    try:
        # Initialize BigQuery client
        bq_client = get_bigquery_client()

        # Query all available data (no date limit initially)
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
            return pd.DataFrame(), "BigQuery (no data)"

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
        st.warning(f"Falling back to Alpha Vantage for {ticker}")

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
    try:
        bq_client = get_bigquery_client()
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
