"""
Price Chart Module

This module provides functions for creating the main price chart with
technical indicators, SMAs, Bollinger Bands, and forecast data.

Functions:
    create_price_chart(data): Create the main price chart with indicators
    add_sma_traces(fig, data): Add Simple Moving Average traces
    add_bollinger_bands(fig, data): Add Bollinger Band traces
    add_cross_signals(fig, data): Add golden/death cross signals
    add_forecast_traces(fig, data): Add forecast and confidence intervals

Author: Shane
Created: 2025-09-02 (Refactored from plotter.py)
"""

import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from typing import Optional


def create_price_chart(
    data: pd.DataFrame, ticker_name: str, selected_stock: str
) -> go.Figure:
    """
    Create the main price chart with all technical indicators.

    Args:
        data: Stock data DataFrame with price and indicator columns
        ticker_name: Display name for the ticker
        selected_stock: Stock symbol

    Returns:
        Plotly figure with price chart and indicators
    """
    fig = go.Figure()

    # Add all price-related traces
    add_cross_signals(fig, data)
    add_bollinger_bands(fig, data)
    add_sma_traces(fig, data)
    add_forecast_traces(fig, data)
    add_price_trace(fig, data)

    # Update layout
    fig.update_layout(
        title=f"Price & Forecast for {ticker_name} ({selected_stock})",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        height=400,
    )

    return fig


def add_cross_signals(fig: go.Figure, data: pd.DataFrame) -> None:
    """
    Add golden cross and death cross signals to the chart.

    Args:
        fig: Plotly figure to add traces to
        data: DataFrame containing cross signal data
    """
    # Death Cross Signal
    if "DeathCross_Signal" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["DeathCross_Signal"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                name="Death Cross (SMA50/200)",
                visible="legendonly",
            )
        )
    else:
        st.warning("Death Cross Signal not found in data. Skipping this trace.")

    # Golden Cross Signal
    if "GoldenCross_Signal" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["GoldenCross_Signal"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="green"),
                name="Golden Cross (SMA50/200)",
                visible="legendonly",
            )
        )
    else:
        st.warning("Golden Cross Signal not found in data. Skipping this trace.")


def add_bollinger_bands(fig: go.Figure, data: pd.DataFrame) -> None:
    """
    Add Bollinger Bands to the chart.

    Args:
        fig: Plotly figure to add traces to
        data: DataFrame containing Bollinger Band data
    """
    # Lower Bollinger Band
    if "bb_lower" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["bb_lower"],
                line=dict(color="green", width=1),
                name="Lower BB",
                visible="legendonly",
            )
        )
    else:
        st.warning("Bollinger Band Lower not found in data. Skipping this trace.")

    # Upper Bollinger Band
    if "bb_upper" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["bb_upper"],
                line=dict(color="red", width=1),
                name="Upper BB",
                visible="legendonly",
            )
        )
    else:
        st.warning("Bollinger Band Upper not found in data. Skipping this trace.")


def add_sma_traces(fig: go.Figure, data: pd.DataFrame) -> None:
    """
    Add Simple Moving Average traces to the chart.

    Args:
        fig: Plotly figure to add traces to
        data: DataFrame containing SMA data
    """
    sma_configs = [
        {"col": "SMA200", "name": "SMA200", "color": "teal", "width": 1},
        {"col": "SMA100", "name": "SMA100", "color": "purple", "width": 1},
        {"col": "SMA50", "name": "SMA50", "color": "black", "width": 2, "dash": "dash"},
        {"col": "SMA20", "name": "SMA20", "color": "black", "width": 1, "dash": "dot"},
    ]

    for config in sma_configs:
        if config["col"] in data.columns:
            line_config = {"color": config["color"], "width": config["width"]}
            if "dash" in config:
                line_config["dash"] = config["dash"]

            fig.add_trace(
                go.Scatter(
                    x=data["date"],
                    y=data[config["col"]],
                    name=config["name"],
                    line=line_config,
                    visible="legendonly",
                )
            )
        else:
            st.warning(f"{config['name']} not found in data. Skipping this trace.")


def add_forecast_traces(fig: go.Figure, data: pd.DataFrame) -> None:
    """
    Add forecast and confidence interval traces to the chart.

    Args:
        fig: Plotly figure to add traces to
        data: DataFrame containing forecast data
    """
    # Determine the date column name - check all possible variants
    date_col = None
    for col_name in ["ds", "date", "Date"]:
        if col_name in data.columns:
            date_col = col_name
            break
    
    # If no date column found, skip forecast traces
    if date_col is None:
        return
    
    # Forecast confidence interval
    if "yhat_lower" in data.columns and "yhat_upper" in data.columns:
        # Lower bound
        fig.add_trace(
            go.Scatter(
                x=data[date_col],
                y=data["yhat_lower"],
                line=dict(color="lightblue", width=0),
                name="Forecast Lower Bound",
                showlegend=False,
            )
        )

        # Upper bound with fill
        fig.add_trace(
            go.Scatter(
                x=data[date_col],
                y=data["yhat_upper"],
                line=dict(color="lightblue", width=0),
                name="Forecast Upper Bound",
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.4)",
            )
        )
    else:
        st.warning(
            "Forecast bounds (yhat_lower/yhat_upper) not found in data. Skipping these traces."
        )

    # Main forecast line
    if "yhat" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data[date_col],
                y=data["yhat"],
                line=dict(color="blue", width=2),
                name="Forecast",
                mode="lines",
            )
        )
    else:
        st.warning("Forecast (yhat) not found in data. Skipping this trace.")


def add_price_trace(fig: go.Figure, data: pd.DataFrame) -> None:
    """
    Add the actual price trace to the chart.

    Args:
        fig: Plotly figure to add traces to
        data: DataFrame containing price data
    """
    # Determine price column - using snake_case internally
    if "close" in data.columns:
        price_col = "close"
    elif "adjusted_close" in data.columns:
        price_col = "adjusted_close"
    else:
        price_col = "close"  # Default fallback

    if price_col in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data[price_col],
                mode="lines",
                name="Close Price",
                line=dict(color="orange", width=2),
            )
        )
    else:
        st.warning(f"{price_col} not found in data. Skipping this trace.")


def configure_price_chart_layout(fig: go.Figure, data: pd.DataFrame) -> None:
    """
    Configure the layout and display range for the price chart.

    Args:
        fig: Plotly figure to configure
        data: DataFrame to calculate display range from
    """
    # Set initial range to display approx 1.5 years by default
    end_date = data["date"].max()
    display_period_days = min(len(data), 365 * 1.5)
    start_date = end_date - pd.Timedelta(days=display_period_days)
    start_date = max(start_date, data["date"].min())

    fig.update_xaxes(range=[start_date, end_date])
    fig.update_xaxes(rangeslider_visible=True)
