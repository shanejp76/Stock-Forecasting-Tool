"""
Technical Indicators Module

This module provides functions for calculating technical indicators
including Bollinger Bands, RSI, MACD, and cross signals.

Functions:
    process_technical_indicators(data): Add all technical indicators to data
    calculate_moving_averages(data): Calculate SMA indicators and cross signals
    calculate_cross_signals(data): Calculate Death Cross and Golden Cross signals
    calculate_bollinger_bands(data): Calculate Bollinger Bands
    calculate_rsi(data): Calculate RSI indicator
    calculate_macd(data): Calculate MACD indicator
    determine_periods(data, volatility, user_training_days): Determine optimal periods

Author: Shane
Created: 2025-09-02 (Refactored from data_handler.py)
Updated: 2025-09-07 (Added Death Cross and Golden Cross signals)
"""

import pandas as pd
import numpy as np
import ta
from typing import Optional, Tuple


def process_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators to stock data.

    Args:
        data: Stock data DataFrame with OHLCV columns

    Returns:
        DataFrame with technical indicators added
    """
    if data.empty or "close" not in data.columns:
        return data

    # Make a copy to avoid modifying original data
    data_with_indicators = data.copy()

    # Calculate moving averages
    data_with_indicators = calculate_moving_averages(data_with_indicators)

    # Calculate Bollinger Bands
    data_with_indicators = calculate_bollinger_bands(data_with_indicators)

    # Calculate RSI
    data_with_indicators = calculate_rsi(data_with_indicators)

    # Calculate MACD
    data_with_indicators = calculate_macd(data_with_indicators)

    # Calculate additional indicators
    data_with_indicators = calculate_additional_indicators(data_with_indicators)

    return data_with_indicators


def calculate_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages (SMA) and cross signals.

    Args:
        data: Stock data DataFrame

    Returns:
        DataFrame with SMA columns and cross signals added
    """
    if "close" not in data.columns:
        return data

    # Calculate SMAs for different periods
    data["SMA_20"] = ta.trend.sma_indicator(data["close"], window=20)
    data["SMA_50"] = ta.trend.sma_indicator(data["close"], window=50)
    data["SMA_100"] = ta.trend.sma_indicator(data["close"], window=100)
    data["SMA_200"] = ta.trend.sma_indicator(data["close"], window=200)

    # Calculate Death Cross and Golden Cross signals
    data = calculate_cross_signals(data)

    return data


def calculate_cross_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Death Cross and Golden Cross signals.

    Death Cross: SMA50 crosses below SMA200 (bearish signal)
    Golden Cross: SMA50 crosses above SMA200 (bullish signal)

    Args:
        data: DataFrame with SMA_50 and SMA_200 columns

    Returns:
        DataFrame with DeathCross_Signal and GoldenCross_Signal columns added
    """
    if not all(col in data.columns for col in ["SMA_50", "SMA_200"]):
        # Initialize empty signal columns if SMAs don't exist
        data["DeathCross_Signal"] = np.nan
        data["GoldenCross_Signal"] = np.nan
        return data

    # Initialize signal columns with NaN
    data["DeathCross_Signal"] = np.nan
    data["GoldenCross_Signal"] = np.nan

    # Calculate crossover signals
    # Previous period comparison (SMA50 was above/below SMA200)
    sma50_above_sma200_prev = data["SMA_50"].shift(1) > data["SMA_200"].shift(1)
    sma50_below_sma200_prev = data["SMA_50"].shift(1) < data["SMA_200"].shift(1)

    # Current period comparison
    sma50_above_sma200_curr = data["SMA_50"] > data["SMA_200"]
    sma50_below_sma200_curr = data["SMA_50"] < data["SMA_200"]

    # Death Cross: SMA50 was above SMA200, now below
    death_cross_mask = sma50_above_sma200_prev & sma50_below_sma200_curr

    # Golden Cross: SMA50 was below SMA200, now above
    golden_cross_mask = sma50_below_sma200_prev & sma50_above_sma200_curr

    # Set signal values to close price where crossover occurs
    data.loc[death_cross_mask, "DeathCross_Signal"] = data.loc[
        death_cross_mask, "close"
    ]
    data.loc[golden_cross_mask, "GoldenCross_Signal"] = data.loc[
        golden_cross_mask, "close"
    ]

    return data


def calculate_bollinger_bands(
    data: pd.DataFrame, window: int = 20, std_dev: int = 2
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        data: Stock data DataFrame
        window: Period for moving average calculation
        std_dev: Number of standard deviations for bands

    Returns:
        DataFrame with Bollinger Band columns added
    """
    if "close" not in data.columns:
        return data

    # Calculate Bollinger Bands using ta library
    bb_high = ta.volatility.bollinger_hband(
        data["close"], window=window, window_dev=std_dev
    )
    bb_low = ta.volatility.bollinger_lband(
        data["close"], window=window, window_dev=std_dev
    )
    bb_mid = ta.volatility.bollinger_mavg(data["close"], window=window)

    data["bb_upper"] = bb_high
    data["bb_lower"] = bb_low
    data["Bollinger_Middle"] = bb_mid

    # Calculate Bollinger Band width and position
    data["BB_Width"] = (bb_high - bb_low) / bb_mid
    data["BB_Position"] = (data["close"] - bb_low) / (bb_high - bb_low)

    return data


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data: Stock data DataFrame
        window: Period for RSI calculation

    Returns:
        DataFrame with RSI column added
    """
    if "close" not in data.columns:
        return data

    # Calculate RSI using ta library
    data["RSI"] = ta.momentum.rsi(data["close"], window=window)

    # Add RSI signal classifications
    data["RSI_Signal"] = "Neutral"
    data.loc[data["RSI"] > 70, "RSI_Signal"] = "Overbought"
    data.loc[data["RSI"] < 30, "RSI_Signal"] = "Oversold"

    return data
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data: Stock data DataFrame
        window: Period for RSI calculation

    Returns:
        DataFrame with RSI column added
    """
    if "close" not in data.columns:
        return data

    # Calculate RSI using ta library
    data["RSI"] = ta.momentum.rsi(data["close"], window=window)

    # Add RSI signal classifications
    data["RSI_Signal"] = "Neutral"
    data.loc[data["RSI"] > 70, "RSI_Signal"] = "Overbought"
    data.loc[data["RSI"] < 30, "RSI_Signal"] = "Oversold"

    return data


def calculate_macd(
    data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        data: Stock data DataFrame
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period

    Returns:
        DataFrame with MACD columns added
    """
    if "close" not in data.columns:
        return data

    # Calculate MACD components using ta library
    data["MACD"] = ta.trend.macd(data["close"], window_fast=fast, window_slow=slow)
    data["MACD_Signal"] = ta.trend.macd_signal(
        data["close"], window_fast=fast, window_slow=slow, window_sign=signal
    )
    data["MACD_Hist"] = ta.trend.macd_diff(
        data["close"], window_fast=fast, window_slow=slow, window_sign=signal
    )

    # Add MACD signal classifications
    data["MACD_Signal_Direction"] = "Neutral"
    data.loc[data["MACD"] > data["MACD_Signal"], "MACD_Signal_Direction"] = "Bullish"
    data.loc[data["MACD"] < data["MACD_Signal"], "MACD_Signal_Direction"] = "Bearish"

    return data


def calculate_additional_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional technical indicators.

    Args:
        data: Stock data DataFrame

    Returns:
        DataFrame with additional indicators added
    """
    if (
        "close" not in data.columns
        or "high" not in data.columns
        or "low" not in data.columns
    ):
        return data

    # Stochastic Oscillator
    data["Stoch_K"] = ta.momentum.stoch(data["high"], data["low"], data["close"])
    data["Stoch_D"] = ta.momentum.stoch_signal(data["high"], data["low"], data["close"])

    # Average True Range (ATR)
    data["ATR"] = ta.volatility.average_true_range(
        data["high"], data["low"], data["close"]
    )

    # Commodity Channel Index (CCI)
    data["CCI"] = ta.trend.cci(data["high"], data["low"], data["close"])

    # Williams %R
    data["Williams_R"] = ta.momentum.williams_r(
        data["high"], data["low"], data["close"]
    )

    return data

    return data


def determine_periods(
    data: pd.DataFrame, volatility: float, user_training_days: Optional[int] = None
) -> Tuple[int, int, int]:
    """
    Determine optimal training and forecast periods based on data characteristics.

    Args:
        data: Stock data DataFrame
        volatility: Calculated volatility measure
        user_training_days: User-specified training days (overrides calculation if provided)

    Returns:
        Tuple of (period_unit, forecast_period, training_period)
    """
    data_length = len(data)

    # Calculate period_unit for cross-validation
    period_unit = 365 if data_length >= 504 else data_length // 4

    # Default periods
    default_training = min(500, data_length)  # Max 500 days or available data
    default_forecast = 30  # 30 days default

    # If user specified training days, use those
    if user_training_days is not None:
        training_period = min(user_training_days, data_length)
    else:
        # Adjust based on volatility
        if volatility < 0.15:  # Low volatility
            training_period = min(400, data_length)
            forecast_period = 45
        elif volatility < 0.25:  # Moderate volatility
            training_period = min(450, data_length)
            forecast_period = 30
        elif volatility < 0.40:  # High volatility
            training_period = min(350, data_length)
            forecast_period = 21
        else:  # Very high volatility
            training_period = min(300, data_length)
            forecast_period = 14

        # Ensure minimum training period
        training_period = max(100, training_period)
        return period_unit, forecast_period, training_period

    # For user-specified training days, adjust forecast based on volatility
    if volatility < 0.15:
        forecast_period = 45
    elif volatility < 0.25:
        forecast_period = 30
    elif volatility < 0.40:
        forecast_period = 21
    else:
        forecast_period = 14

    return period_unit, forecast_period, training_period


def get_technical_summary(data: pd.DataFrame) -> dict:
    """
    Generate a summary of technical indicator signals.

    Args:
        data: DataFrame with technical indicators

    Returns:
        Dictionary with technical analysis summary
    """
    if data.empty:
        return {}

    latest = data.iloc[-1]
    summary = {}

    # RSI signal
    if "RSI" in data.columns:
        rsi_value = latest["RSI"]
        if pd.notna(rsi_value):
            if rsi_value > 70:
                summary["RSI"] = {
                    "value": rsi_value,
                    "signal": "Overbought",
                    "action": "Sell",
                }
            elif rsi_value < 30:
                summary["RSI"] = {
                    "value": rsi_value,
                    "signal": "Oversold",
                    "action": "Buy",
                }
            else:
                summary["RSI"] = {
                    "value": rsi_value,
                    "signal": "Neutral",
                    "action": "Hold",
                }

    # MACD signal
    if all(col in data.columns for col in ["MACD", "MACD_Signal"]):
        macd_value = latest["MACD"]
        macd_signal = latest["MACD_Signal"]
        if pd.notna(macd_value) and pd.notna(macd_signal):
            if macd_value > macd_signal:
                summary["MACD"] = {"signal": "Bullish", "action": "Buy"}
            else:
                summary["MACD"] = {"signal": "Bearish", "action": "Sell"}

    # Moving Average signals
    if (
        all(col in data.columns for col in ["SMA_20", "SMA_50"])
        and "close" in data.columns
    ):
        price = latest["close"]
        sma_20 = latest["SMA_20"]
        sma_50 = latest["SMA_50"]

        if pd.notna(price) and pd.notna(sma_20) and pd.notna(sma_50):
            if price > sma_20 > sma_50:
                summary["Moving_Averages"] = {
                    "signal": "Strong Bullish",
                    "action": "Buy",
                }
            elif price > sma_20:
                summary["Moving_Averages"] = {"signal": "Bullish", "action": "Buy"}
            elif price < sma_20 < sma_50:
                summary["Moving_Averages"] = {
                    "signal": "Strong Bearish",
                    "action": "Sell",
                }
            else:
                summary["Moving_Averages"] = {"signal": "Bearish", "action": "Sell"}

    return summary
