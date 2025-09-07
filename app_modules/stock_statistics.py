"""
Stock Statistics Module

This module provides functions for calculating stock statistics,
including price statistics, volatility measures, and percentiles.

Functions:
    calculate_stock_stats(data, selected_stock, price_col): Calculate comprehensive stock statistics
    calculate_volatility_metrics(data, price_col): Calculate various volatility measures
    calculate_price_statistics(data, price_col): Calculate basic price statistics
    get_stock_percentiles(data, price_col): Calculate price percentiles

Author: Shane
Created: 2025-09-02 (Refactored from data_handler.py)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def calculate_stock_stats(
    data: pd.DataFrame, selected_stock: str, price_col: str = "Close"
) -> Tuple[Dict[str, Any], Dict[str, float], float]:
    """
    Calculate comprehensive stock statistics including price stats, percentiles, and volatility.

    Args:
        data: Stock data DataFrame with price columns
        selected_stock: Stock symbol for reference
        price_col: Column name to use for price calculations

    Returns:
        Tuple of (stats_dict, percentiles_dict, volatility_float)
    """
    if data.empty or price_col not in data.columns:
        return {}, {}, 0.0

    # Calculate basic price statistics
    price_stats = calculate_price_statistics(data, price_col)

    # Calculate volatility metrics
    volatility_metrics = calculate_volatility_metrics(data, price_col)

    # Calculate percentiles
    percentiles = get_stock_percentiles(data, price_col)

    # Calculate volatility-based winsorization percentiles (tuple format)
    winsorization_percentiles = get_volatility_based_percentiles(
        volatility_metrics.get("Annualized Volatility", 0.0)
    )

    # Combine all statistics
    stats = {
        **price_stats,
        **volatility_metrics,
        "Symbol": selected_stock,
        "data_points": len(data),
        "date_range": f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
    }

    return (
        stats,
        winsorization_percentiles,
        volatility_metrics.get("Annualized Volatility", 0.0),
    )


def calculate_price_statistics(data: pd.DataFrame, price_col: str) -> Dict[str, Any]:
    """
    Calculate basic price statistics.

    Args:
        data: Stock data DataFrame
        price_col: Column name for price data

    Returns:
        Dictionary with price statistics
    """
    if data.empty or price_col not in data.columns:
        return {}

    prices = data[price_col]

    # Basic statistics
    current_price = prices.iloc[-1] if len(prices) > 0 else np.nan
    current_volume = (
        data["volume"].iloc[-1]
        if len(data) > 0 and "volume" in data.columns
        else np.nan
    )
    price_change = prices.iloc[-1] - prices.iloc[0] if len(prices) > 1 else 0
    price_change_pct = (
        (price_change / prices.iloc[0] * 100)
        if len(prices) > 1 and prices.iloc[0] != 0
        else 0
    )

    return {
        "Current Price": current_price,
        "Current Volume": current_volume,
        "price_change": price_change,
        "Daily % Change": price_change_pct,
        "52-Week Low": prices.min(),
        "52-Week High": prices.max(),
        "mean_price": prices.mean(),
        "median_price": prices.median(),
        "price_std": prices.std(),
    }


def calculate_volatility_metrics(
    data: pd.DataFrame, price_col: str
) -> Dict[str, float]:
    """
    Calculate various volatility measures.

    Args:
        data: Stock data DataFrame
        price_col: Column name for price data

    Returns:
        Dictionary with volatility metrics
    """
    if data.empty or price_col not in data.columns:
        return {"volatility": 0.0}

    prices = data[price_col]

    # Calculate daily returns
    returns = prices.pct_change().dropna()

    if len(returns) == 0:
        return {"volatility": 0.0}

    # Annualized volatility (assuming 252 trading days per year)
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)

    # Additional volatility measures
    volatility_metrics = {
        "Annualized Volatility": annualized_volatility,
        "daily_volatility": daily_volatility,
        "returns_mean": returns.mean(),
        "returns_std": returns.std(),
        "returns_skew": returns.skew() if len(returns) > 2 else 0.0,
        "returns_kurtosis": returns.kurtosis() if len(returns) > 3 else 0.0,
        "max_daily_gain": returns.max(),
        "max_daily_loss": returns.min(),
        "positive_days_ratio": (returns > 0).mean(),
    }

    # Value at Risk (95% confidence)
    if len(returns) >= 20:
        var_95 = returns.quantile(0.05)
        volatility_metrics["var_95"] = var_95

    return volatility_metrics


def get_stock_percentiles(data: pd.DataFrame, price_col: str) -> Dict[str, float]:
    """
    Calculate price percentiles for winsorization and analysis.

    Args:
        data: Stock data DataFrame
        price_col: Column name for price data

    Returns:
        Dictionary with percentile values
    """
    if data.empty or price_col not in data.columns:
        return {}

    prices = data[price_col]

    percentiles = {}
    percentile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    for p in percentile_levels:
        percentiles[f"p{p}"] = prices.quantile(p / 100)

    # Add interquartile range
    percentiles["iqr"] = percentiles["p75"] - percentiles["p25"]

    return percentiles


def get_volatility_based_percentiles(volatility: float) -> Tuple[float, float]:
    """
    Get winsorization percentiles based on volatility level.

    Args:
        volatility: Stock volatility measure

    Returns:
        Tuple of (lower_percentile, upper_percentile) for winsorization
    """
    if volatility < 0.15:  # Low volatility
        return (0.15, 0.85)
    elif volatility < 0.25:  # Moderate volatility
        return (0.1, 0.9)
    elif volatility < 0.40:  # High volatility
        return (0.1, 0.9)
    else:  # Very high volatility
        return (0.05, 0.95)


def classify_volatility(volatility: float) -> str:
    """
    Classify volatility into categories.

    Args:
        volatility: Annualized volatility value

    Returns:
        String classification of volatility level
    """
    if volatility < 0.15:
        return "Low"
    elif volatility < 0.25:
        return "Moderate"
    elif volatility < 0.40:
        return "High"
    else:
        return "Very High"


def get_risk_metrics(
    data: pd.DataFrame, price_col: str, benchmark_data: pd.DataFrame = None
) -> Dict[str, float]:
    """
    Calculate risk metrics including Sharpe ratio, beta, etc.

    Args:
        data: Stock data DataFrame
        price_col: Column name for price data
        benchmark_data: Optional benchmark data for beta calculation

    Returns:
        Dictionary with risk metrics
    """
    if data.empty or price_col not in data.columns:
        return {}

    returns = data[price_col].pct_change().dropna()

    if len(returns) == 0:
        return {}

    risk_metrics = {}

    # Sharpe ratio (assuming risk-free rate of 2% annually)
    risk_free_rate = 0.02 / 252  # Daily risk-free rate
    excess_returns = returns - risk_free_rate

    if returns.std() != 0:
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        risk_metrics["sharpe_ratio"] = sharpe_ratio

    # Maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    risk_metrics["max_drawdown"] = max_drawdown

    # Beta calculation (if benchmark provided)
    if benchmark_data is not None and price_col in benchmark_data.columns:
        benchmark_returns = benchmark_data[price_col].pct_change().dropna()

        # Align dates
        aligned_data = pd.DataFrame(
            {"stock": returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned_data) > 1:
            covariance = aligned_data["stock"].cov(aligned_data["benchmark"])
            benchmark_variance = aligned_data["benchmark"].var()

            if benchmark_variance != 0:
                beta = covariance / benchmark_variance
                risk_metrics["beta"] = beta

    return risk_metrics
