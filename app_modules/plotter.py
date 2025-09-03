"""
Plotter Module for Swing Ticker

This module provides the main plotting interface for backward compatibility.
The actual plotting functionality has been refactored into modular components:
- chart_layout.py: Multi-panel chart creation and layout
- price_charts.py: Price chart with technical indicators
- indicator_charts.py: RSI and MACD indicator charts

Functions:
    plot_forecast(data_to_plot, ticker_name_for_plot, selected_stock_for_plot):
        Main plotting function that delegates to the modular chart system.

Author: Shane
Created: 2024-12-04
Refactored: 2025-09-02
"""

# Import the main function from chart_layout module
from .chart_layout import plot_forecast

# Export the main function for backward compatibility
__all__ = ["plot_forecast"]
