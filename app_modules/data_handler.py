"""
Data Handler Module for Swing Ticker

This module provides functions for loading, processing, and analyzing stock data.
It includes utilities for fetching tickers, loading historical price data, calculating
key statistics, and generating technical indicators such as SMA, Bollinger Bands, RSI,
and MACD. It also determines appropriate training and forecast periods based on data
length and volatility.

Functions:
    load_finnhub_tickers(finnhub_api_key, exchange_code): Loads ticker symbols from Finnhub API.
    load_alpha_vantage_data(_ts_av, ticker): Loads daily stock data from Alpha Vantage.
    process_stock_data(data, start_date): Filters and processes raw stock data.
    calculate_stock_stats(data, selected_stock, price_col): Calculates key statistics and volatility.
    process_technical_indicators(data, price_col): Adds technical indicators to the data.
    determine_periods(data, volatility, user_training_days=None): Determines training and forecast periods.

Author: Shane
Created: 2024-12-04
"""

# app_modules/data_handler.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from alpha_vantage.timeseries import TimeSeries
import ta  # This is already imported and will be used for MACD and RSI
from datetime import date, timedelta


@st.cache_data
def load_finnhub_tickers(finnhub_api_key, exchange_code):
    """
    Loads stock ticker symbols and descriptions from Finnhub API.
    Uses a default list if API key is not set or an error occurs.
    """
    tickers = []
    tickers_data = []
    if finnhub_api_key != "YOUR_FINNHUB_API_KEY" and finnhub_api_key != "":
        url = f"https://finnhub.io/api/v1/stock/symbol?exchange={exchange_code}&token={finnhub_api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            tickers_data = response.json()
            tickers = [item["symbol"] for item in tickers_data]
        except requests.exceptions.RequestException as e:
            st.warning(
                f"-- Error fetching Finnhub data: {e}. Ticker search functionality might be limited. --"
            )
            tickers = ["IBM", "GOOG", "MSFT", "AAPL"]
            tickers_data = [
                {"symbol": t, "description": f"{t} Company"} for t in tickers
            ]
    else:
        st.warning(
            "Finnhub API key not set or is placeholder. Ticker search will use default list."
        )
        tickers = ["IBM", "GOOG", "MSFT", "AAPL"]
        tickers_data = [{"symbol": t, "description": f"{t} Company"} for t in tickers]
    return tickers, tickers_data


@st.cache_data
def load_alpha_vantage_data(_ts_av: TimeSeries, ticker: str) -> pd.DataFrame:
    """
    Loads daily stock data from Alpha Vantage.
    The `_ts_av` parameter is prefixed with an underscore to prevent Streamlit from hashing it,
    as TimeSeries objects are not hashable.
    """
    try:
        data, meta_data = _ts_av.get_daily(ticker, outputsize="full")
        data.columns = ["Open", "High", "Low", "Close", "Volume"]
        data.index.name = "Date"
        data.reset_index(inplace=True)
        # Convert to datetime objects, not just date objects
        # This ensures the 'Date' column is a proper datetime64[ns] dtype
        data["Date"] = pd.to_datetime(data["Date"])
        data["Adjusted Close"] = data["Close"]
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        st.info(
            f"This often happens if you've hit API rate limits. Please try again tomorrow."
        )
        return pd.DataFrame()


def process_stock_data(data, start_date):
    """
    Processes and filters the raw stock data.
    """
    if data.empty:
        return pd.DataFrame()

    # Ensure data['Date'] is already datetime64[ns] from load_alpha_vantage_data
    # Convert start_date (which is a datetime.date object) to a pandas datetime object
    # to ensure consistent data types for comparison.
    start_date_ts = pd.to_datetime(start_date)

    data = data[::-1].reset_index(drop=True)
    # Perform the comparison with the pandas datetime object
    data = data[data["Date"] >= start_date_ts].reset_index(drop=True)
    return data


def calculate_stock_stats(data, selected_stock, price_col):
    """
    Calculates key statistics for the selected stock.
    Returns stats dictionary, percentile tuple, and volatility.
    """
    stats = {}
    percentiles = (0.05, 0.95)  # Default
    volatility = 0.0

    if not data.empty and price_col in data.columns:
        stats["Symbol"] = selected_stock
        stats["Current Price"] = f"${round(data[price_col].iloc[-1], 2):,.2f}"
        stats["Current Volume"] = f"{int(data['Volume'].iloc[-1]):,}"

        # Daily Percentage Change
        if len(data) > 1:
            daily_change = (data[price_col].iloc[-1] - data[price_col].iloc[-2]) / data[
                price_col
            ].iloc[-2]
            stats["Daily % Change"] = f"{daily_change * 100:.2f}%"
        else:
            stats["Daily % Change"] = "N/A"  # Not enough data for daily change

        # Year-to-Date (YTD) Percentage Change
        # Find the first trading day of the current year in the data
        current_year = data["Date"].iloc[-1].year
        ytd_start_price_series = data[data["Date"].dt.year == current_year][price_col]

        if not ytd_start_price_series.empty:
            ytd_start_price = ytd_start_price_series.iloc[0]
            ytd_change = (data[price_col].iloc[-1] - ytd_start_price) / ytd_start_price
            stats["YTD % Change"] = f"{ytd_change * 100:.2f}%"
        else:
            stats["YTD % Change"] = "N/A"  # No data for current year

        # 52-Week High/Low
        # Filter data for the last 52 weeks (approx 365 days, handling non-trading days)
        one_year_ago = data["Date"].iloc[-1] - pd.Timedelta(days=365)
        last_52_weeks_data = data[data["Date"] >= one_year_ago]

        if not last_52_weeks_data.empty:
            stats["52-Week High"] = (
                f"${round(last_52_weeks_data[price_col].max(), 2):,.2f}"
            )
            stats["52-Week Low"] = (
                f"${round(last_52_weeks_data[price_col].min(), 2):,.2f}"
            )
        else:
            stats["52-Week High"] = "N/A"
            stats["52-Week Low"] = "N/A"

        # Last Update / Data Freshness
        stats["Last Data Date"] = data["Date"].iloc[-1].strftime("%Y-%m-%d")

        # Earliest date available in the data (after filtering for DYNAMIC_START_DATE)
        stats["Earliest Data Date"] = data["Date"].iloc[0].strftime("%Y-%m-%d")

        data["daily_returns"] = data[price_col].pct_change()
        volatility = data["daily_returns"].std() * np.sqrt(252)

        if volatility < 0.2:
            category = "Low"
            percentiles = (0.15, 0.85)
        elif volatility < 0.4:
            category = "Medium-Low"
            percentiles = (0.1, 0.9)
        elif volatility < 0.6:
            category = "Medium"
            percentiles = (0.1, 0.9)
        elif volatility < 0.8:
            category = "Medium-High"
            percentiles = (0.05, 0.95)
        else:
            category = "High"
            percentiles = (0.05, 0.95)
        stats["Annualized Volatility"] = category
        stats["Average Daily Percentage Change"] = (
            str(round(data["daily_returns"].mean() * 100, 4)) + " %"
        )
    else:
        st.warning(
            "Could not retrieve stock statistics. Please ensure the symbol is valid and data loaded correctly."
        )
        # It's important to still return something, even if empty, to avoid further errors down the line
        return {}, (0.05, 0.95), 0.0  # Return default values
    return stats, percentiles, volatility


def process_technical_indicators(data, price_col):
    """
    Calculates technical indicators like SMA, Bollinger Bands, RSI, and MACD.
    """
    if not data.empty and price_col in data.columns:
        # Existing SMA50 and Bollinger Bands
        data["SMA50"] = data[price_col].rolling(window=50).mean()
        indicator_bb = ta.volatility.BollingerBands(
            close=data[price_col], window=20, window_dev=2
        )
        data["bb_upper"] = indicator_bb.bollinger_hband()
        data["bb_lower"] = indicator_bb.bollinger_lband()

        # Add other relevant SMAs
        data["SMA20"] = data[price_col].rolling(window=20).mean()
        data["SMA100"] = data[price_col].rolling(window=100).mean()
        data["SMA200"] = data[price_col].rolling(window=200).mean()

        # Calculate Golden Cross (SMA50 crosses above SMA200)
        # Check if SMA50 was below SMA200 in the previous period and is above in the current period
        data["GoldenCross"] = (data["SMA50"].shift(1) < data["SMA200"].shift(1)) & (
            data["SMA50"] > data["SMA200"]
        )
        # Get the 'Adjusted Close' price at the point of the Golden Cross
        data["GoldenCross_Signal"] = np.where(
            data["GoldenCross"], data[price_col], np.nan
        )

        # Calculate Death Cross (SMA50 crosses below SMA200)
        # Check if SMA50 was above SMA200 in the previous period and is below in the current period
        data["DeathCross"] = (data["SMA50"].shift(1) > data["SMA200"].shift(1)) & (
            data["SMA50"] < data["SMA200"]
        )
        # Get the 'Adjusted Close' price at the point of the Death Cross
        data["DeathCross_Signal"] = np.where(
            data["DeathCross"], data[price_col], np.nan
        )

        # --- ADD NEW INDICATORS: RSI and MACD ---
        # RSI (Relative Strength Index) - CORRECTED FUNCTION CALL
        data["RSI"] = ta.momentum.rsi(
            close=data[price_col], window=14
        )  # Common window is 14 periods

        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(
            close=data[price_col],
            window_fast=12,  # Common fast window is 12 periods
            window_slow=26,  # Common slow window is 26 periods
            window_sign=9,  # Common signal window is 9 periods
        )
        data["MACD"] = macd.macd()
        data["MACD_Signal"] = macd.macd_signal()
        data["MACD_Hist"] = macd.macd_diff()  # MACD Histogram

    else:
        st.error(
            "Cannot process indicators: 'Adjusted Close' column not found or data is empty."
        )
        st.stop()
    return data


def determine_periods(data, volatility, user_training_days=None):
    """
    Determines training and forecast periods based on stock data length and volatility.
    If user_training_days is provided, uses that value instead of automatic calculation.
    """
    if not data.empty:
        data_len_years = len(data) / 365
        if data_len_years < 2:
            period_unit = int(len(data) / 4)
            forecast_period = period_unit
            # Use user-defined training days if provided, otherwise use all available data
            train_period = user_training_days if user_training_days else len(data)
        else:
            period_unit = 365
            forecast_period = period_unit
            if user_training_days:
                # Use user-defined training days
                train_period = user_training_days
            else:
                # Use automatic calculation based on volatility
                train_period = (
                    forecast_period * 4 if volatility < 0.6 else forecast_period * 8
                )

        # Ensure train_period doesn't exceed available data
        max_available_days = len(data)
        if train_period > max_available_days:
            train_period = max_available_days

        # Ensure these are always integers before being passed around
        period_unit = int(period_unit)
        forecast_period = int(forecast_period)
        train_period = int(train_period)
    else:
        st.warning("Data is empty, cannot determine stock age or set training periods.")
        st.stop()
    return period_unit, forecast_period, train_period
