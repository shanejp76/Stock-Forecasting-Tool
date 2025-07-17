# app_modules/market_correlation.py
import pandas as pd
import numpy as np
import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from app_modules.data_handler import load_alpha_vantage_data  # We'll need this function


def calculate_market_correlation(
    _ts_av: TimeSeries, stock_data: pd.DataFrame, market_ticker: str = "SPY"
) -> float:
    """
    Calculates the correlation between the selected stock and a market index (e.g., S&P 500).

    Args:
        _ts_av: Initialized Alpha Vantage TimeSeries object.
        stock_data: DataFrame containing historical data for the selected stock,
                    must include 'Date' and 'Adjusted Close' columns.
        market_ticker: Ticker symbol for the market index (default is "SPY").

    Returns:
        The correlation coefficient as a float, or None if data is insufficient.
    """
    if stock_data.empty or "Adjusted Close" not in stock_data.columns:
        st.warning(
            "Stock data is empty or 'Adjusted Close' column missing for correlation calculation."
        )
        return None

    # Load market index data
    market_data = load_alpha_vantage_data(_ts_av, market_ticker)

    if market_data.empty or "Adjusted Close" not in market_data.columns:
        st.warning(
            f"Could not load market data for {market_ticker} for correlation calculation."
        )
        return None

    # Merge dataframes on 'Date'
    # Ensure 'Date' columns are datetime objects for proper merging
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    market_data["Date"] = pd.to_datetime(market_data["Date"])

    merged_data = pd.merge(
        stock_data[["Date", "Adjusted Close"]],
        market_data[["Date", "Adjusted Close"]],
        on="Date",
        how="inner",
        suffixes=("_stock", "_market"),
    )

    if merged_data.empty or len(merged_data) < 2:
        st.warning(
            f"Not enough common data points between stock and {market_ticker} for correlation."
        )
        return None

    # Calculate daily returns
    merged_data["stock_returns"] = merged_data["Adjusted Close_stock"].pct_change()
    merged_data["market_returns"] = merged_data["Adjusted Close_market"].pct_change()

    # Drop NaN values from returns (first row will be NaN)
    merged_data.dropna(inplace=True)

    if merged_data.empty:
        st.warning("No overlapping daily returns to calculate correlation.")
        return None

    # Calculate correlation
    correlation = merged_data["stock_returns"].corr(merged_data["market_returns"])

    return correlation
