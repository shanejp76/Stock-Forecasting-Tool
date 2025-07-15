import streamlit as st
import pandas as pd
import numpy as np
import requests
from alpha_vantage.timeseries import TimeSeries
import ta
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
        stats["Current Price"] = round(data[price_col].iloc[-1], 2)
        stats["Current Volume"] = data["Volume"].iloc[-1]
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
    Calculates technical indicators like SMA and Bollinger Bands.
    """
    if not data.empty and price_col in data.columns:
        data["SMA50"] = data[price_col].rolling(window=50).mean()
        indicator_bb = ta.volatility.BollingerBands(
            close=data[price_col], window=20, window_dev=2
        )
        data["bb_upper"] = indicator_bb.bollinger_hband()
        data["bb_lower"] = indicator_bb.bollinger_lband()
    else:
        st.error(
            "Cannot process indicators: 'Adjusted Close' column not found or data is empty."
        )
        st.stop()
    return data


def determine_periods(data, volatility):
    """
    Determines training and forecast periods based on stock data length and volatility.
    """
    if not data.empty:
        data_len_years = len(data) / 365
        if data_len_years < 2:
            period_unit = int(len(data) / 4)
            forecast_period = period_unit
            train_period = len(data)  # Train on all available data
        else:
            period_unit = 365
            forecast_period = period_unit
            train_period = (
                forecast_period * 4 if volatility < 0.6 else forecast_period * 8
            )

        # Ensure these are always integers before being passed around
        period_unit = int(period_unit)
        forecast_period = int(forecast_period)
        train_period = int(train_period)
    else:
        st.warning("Data is empty, cannot determine stock age or set training periods.")
        st.stop()
    return period_unit, forecast_period, train_period
