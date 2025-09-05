"""
Chart Layout Module

This module provides functions for creating multi-panel chart layouts
combining price charts with technical indicator subplots.

Functions:
    create_multi_panel_chart(data, ticker_name, selected_stock): Create complete chart layout
    configure_subplot_layout(fig, data): Configure subplot spacing and ranges
    create_chart_subplots(): Create subplot structure
    apply_chart_styling(fig, ticker_name, selected_stock): Apply consistent styling

Author: Shane
Created: 2025-09-02 (Refactored from plotter.py)
"""

import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from typing import Optional

from .price_charts import (
    add_cross_signals,
    add_bollinger_bands,
    add_sma_traces,
    add_forecast_traces,
    add_price_trace,
)
from .indicator_charts import add_rsi_to_subplot, add_macd_to_subplot


def prepare_data_for_display(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert snake_case column names to Proper Case for chart display.

    Args:
        data: DataFrame with snake_case columns

    Returns:
        DataFrame with proper case columns for visualization
    """
    display_data = data.copy()

    # Mapping from snake_case to Proper Case for display
    column_mapping = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjusted_close": "Adjusted Close",
        "volume": "Volume",
    }

    # Apply mapping only for columns that exist
    for snake_case, proper_case in column_mapping.items():
        if snake_case in display_data.columns:
            display_data = display_data.rename(columns={snake_case: proper_case})

    return display_data


@st.cache_resource
def create_multi_panel_chart(
    data_to_plot: pd.DataFrame, ticker_name_for_plot: str, selected_stock_for_plot: str
) -> None:
    """
    Create a comprehensive multi-panel chart with price, RSI, and MACD.

    Args:
        data_to_plot: DataFrame with price and indicator data
        ticker_name_for_plot: Display name for the ticker
        selected_stock_for_plot: Stock symbol
    """
    if data_to_plot.empty:
        st.error("Cannot plot forecast: data_to_plot is empty.")
        return

    # Use original data directly (snake_case columns)
    display_data = data_to_plot

    # Create subplots: 3 rows, 1 column
    fig = create_chart_subplots(ticker_name_for_plot, selected_stock_for_plot)

    # Add all traces to appropriate subplots
    add_price_traces_to_subplot(fig, display_data)
    add_rsi_to_subplot(fig, display_data, row=2)
    add_macd_to_subplot(fig, display_data, row=3)

    # Configure layout and styling (use original data for layout configuration)
    configure_subplot_layout(fig, data_to_plot)
    apply_chart_styling(fig, ticker_name_for_plot, selected_stock_for_plot)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


def create_chart_subplots(ticker_name: str, selected_stock: str) -> go.Figure:
    """
    Create the subplot structure for the multi-panel chart.

    Args:
        ticker_name: Display name for the ticker
        selected_stock: Stock symbol

    Returns:
        Plotly figure with subplot structure
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],  # 60% price, 20% RSI, 20% MACD
        subplot_titles=(
            f"Price & Forecast for {ticker_name} ({selected_stock})",
            "Relative Strength Index (RSI)",
            "Moving Average Convergence Divergence (MACD)",
        ),
    )
    return fig


def add_price_traces_to_subplot(fig: go.Figure, data: pd.DataFrame) -> None:
    """
    Add all price-related traces to the top subplot.

    Args:
        fig: Plotly figure with subplots
        data: DataFrame containing price and indicator data
    """
    # Add traces in specific order for legend appearance
    add_cross_signals(fig, data)
    add_bollinger_bands(fig, data)
    add_sma_traces(fig, data)
    add_forecast_traces(fig, data)
    add_price_trace(fig, data)


def configure_subplot_layout(fig: go.Figure, data: pd.DataFrame) -> None:
    """
    Configure the layout, axes, and display ranges for all subplots.

    Args:
        fig: Plotly figure to configure
        data: DataFrame to calculate display range from
    """
    # Configure x-axes
    fig.update_xaxes(rangeslider_visible=True, row=3, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)

    # Configure y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    # Set initial display range (approximately 1.5 years)
    end_date = data["date"].max()
    display_period_days = min(len(data), 365 * 1.5)
    start_date = end_date - pd.Timedelta(days=display_period_days)
    start_date = max(start_date, data["date"].min())

    # Apply the range to all x-axes
    for row in [1, 2, 3]:
        fig.update_xaxes(range=[start_date, end_date], row=row, col=1)


def apply_chart_styling(fig: go.Figure, ticker_name: str, selected_stock: str) -> None:
    """
    Apply consistent styling and layout to the chart.

    Args:
        fig: Plotly figure to style
        ticker_name: Display name for the ticker
        selected_stock: Stock symbol
    """
    fig.update_layout(
        title_text=f"Stock Price Forecast and Indicators for {ticker_name} ({selected_stock})",
        height=700,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )


def plot_forecast(
    data_to_plot: pd.DataFrame, ticker_name_for_plot: str, selected_stock_for_plot: str
) -> None:
    """
    Main plotting function - creates the complete forecast visualization.

    This function maintains backward compatibility with the original plotter.py interface.

    Args:
        data_to_plot: DataFrame with price and indicator data
        ticker_name_for_plot: Display name for the ticker
        selected_stock_for_plot: Stock symbol
    """
    create_multi_panel_chart(
        data_to_plot, ticker_name_for_plot, selected_stock_for_plot
    )
