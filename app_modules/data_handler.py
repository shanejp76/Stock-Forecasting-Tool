"""
Data Handler Module for Swing Ticker

This module provides the main data handling interface for backward compatibility.
The actual data handling functionality has been refactored into modular components:
- data_sources.py: Data loading from various APIs
- data_processing.py: Data cleaning and transformation
- stock_statistics.py: Statistical calculations
- technical_indicators.py: Technical indicator calculations

Functions:
    load_stock_data(symbol, start_date, end_date): Load and process stock data
    filter_data_quality(data): Filter out poor quality data
    get_stock_statistics(data): Calculate stock statistics
    process_technical_indicators(data): Add technical indicators

Author: Shane
Created: 2024-12-04
Refactored: 2025-09-02
"""

# Import the main functions from refactored modules
from .data_sources import (
    load_stock_data_hybrid,
    load_stock_data_hybrid as load_stock_data,
    load_finnhub_tickers,
    load_bigquery_data,
    load_bigquery_symbols,
    load_alpha_vantage_data,
)
from .data_processing import (
    process_stock_data,
    clean_stock_data,
)
from .stock_statistics import (
    calculate_stock_stats,
    calculate_volatility_metrics,
    calculate_price_statistics,
    get_stock_percentiles,
    classify_volatility,
    get_risk_metrics,
)
from .technical_indicators import (
    process_technical_indicators,
    determine_periods,
    get_technical_summary,
)

# Export the main functions for backward compatibility
__all__ = [
    "load_stock_data",
    "load_stock_data_hybrid",
    "load_finnhub_tickers",
    "load_bigquery_data",
    "load_bigquery_symbols",
    "load_alpha_vantage_data",
    "process_stock_data",
    "clean_stock_data",
    "calculate_stock_stats",
    "calculate_volatility_metrics",
    "calculate_price_statistics",
    "get_stock_percentiles",
    "classify_volatility",
    "get_risk_metrics",
    "process_technical_indicators",
    "determine_periods",
    "get_technical_summary",
]
