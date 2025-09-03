"""
Market Correlation Module for Swing Ticker

This module provides functionality to calculate the correlation between a selected stock
and a market index (e.g., S&P 500) using historical price data. It helps users assess
how closely the stock's movements align with the broader market, which is useful for
risk management and portfolio diversification analysis.

Functions:
    calculate_market_correlation(_ts_av, stock_data, use_bigquery, market_ticker="SPY"):
        Calculates the correlation coefficient between the selected stock and the market index.

Author: Shane
Created: 2024-12-04
"""

# app_modules/market_correlation.py
import pandas as pd
import numpy as np
import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from app_modules.data_handler import load_alpha_vantage_data, load_stock_data_hybrid
from datetime import date, timedelta


def calculate_market_correlation(
    _ts_av: TimeSeries,
    stock_data: pd.DataFrame,
    use_bigquery: bool = True,
    market_ticker: str = "SPY",
) -> float:
    """
    Calculates the correlation between the selected stock and a market index (e.g., S&P 500).

    Args:
        _ts_av: Initialized Alpha Vantage TimeSeries object.
        stock_data: DataFrame containing historical data for the selected stock,
                    must include 'Date' and 'Close' columns.
        use_bigquery: Boolean flag to determine data source (BigQuery vs Alpha Vantage).
        market_ticker: Ticker symbol for the market index (default is "SPY").

    Returns:
        The correlation coefficient as a float, or None if data is insufficient.
    """
    # Determine the price column to use
    price_col = "Close" if "Close" in stock_data.columns else "close"

    if stock_data.empty or price_col not in stock_data.columns:
        st.warning(
            f"Stock data is empty or '{price_col}' column missing for correlation calculation."
        )
        return None

    # Load market index data using the hybrid approach (respects BigQuery toggle)
    market_data, _ = load_stock_data_hybrid(
        market_ticker,
        date.today() - timedelta(days=2 * 365),  # 2 years back to match stock data
        use_bigquery,
        _ts_av,
    )

    # Determine the price column for market data
    market_price_col = "Close" if "Close" in market_data.columns else "close"

    if market_data.empty or market_price_col not in market_data.columns:
        st.warning(
            f"Could not load market data for {market_ticker} for correlation calculation."
        )
        return None

    # Merge dataframes on date
    # Handle different date column structures (index vs column) - keep snake_case internally
    if stock_data.index.name == "date" or isinstance(
        stock_data.index, pd.DatetimeIndex
    ):
        stock_data = stock_data.reset_index()

    if market_data.index.name == "date" or isinstance(
        market_data.index, pd.DatetimeIndex
    ):
        market_data = market_data.reset_index()

    # Ensure 'date' columns exist and are datetime objects for proper merging
    if "date" not in stock_data.columns:
        stock_data["date"] = stock_data.index
    if "date" not in market_data.columns:
        market_data["date"] = market_data.index

    stock_data["date"] = pd.to_datetime(stock_data["date"])
    market_data["date"] = pd.to_datetime(market_data["date"])

    merged_data = pd.merge(
        stock_data[["date", price_col]],
        market_data[["date", market_price_col]],
        on="date",
        how="inner",
        suffixes=("_stock", "_market"),
    )

    if merged_data.empty or len(merged_data) < 2:
        st.warning(
            f"Not enough common data points between stock and {market_ticker} for correlation."
        )
        return None

    # Calculate daily returns
    merged_data["stock_returns"] = merged_data[f"{price_col}_stock"].pct_change()
    merged_data["market_returns"] = merged_data[
        f"{market_price_col}_market"
    ].pct_change()

    # Drop NaN values from returns (first row will be NaN)
    merged_data.dropna(inplace=True)

    if merged_data.empty:
        st.warning("No overlapping daily returns to calculate correlation.")
        return None

    # Calculate correlation
    correlation = merged_data["stock_returns"].corr(merged_data["market_returns"])

    return correlation
