"""
Indicator Charts Module

This module provides functions for creating technical indicator charts
including RSI and MACD subplots with appropriate styling and ranges.

Functions:
    create_rsi_chart(data): Create RSI chart with overbought/oversold lines
    create_macd_chart(data): Create MACD chart with signal line and histogram
    add_rsi_to_subplot(fig, data, row): Add RSI trace to subplot
    add_macd_to_subplot(fig, data, row): Add MACD traces to subplot

Author: Shane
Created: 2025-09-02 (Refactored from plotter.py)
"""

import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from typing import Optional


def add_rsi_to_subplot(fig: go.Figure, data: pd.DataFrame, row: int = 2) -> None:
    """
    Add RSI indicator to a subplot.

    Args:
        fig: Plotly figure with subplots
        data: DataFrame containing RSI data
        row: Row number for the subplot
    """
    if "RSI" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="purple"),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Add RSI Overbought (70) and Oversold (30) lines
        fig.add_hline(
            y=70,
            line_dash="dot",
            line_color="red",
            annotation_text="Overbought",
            row=row,
            col=1,
        )
        fig.add_hline(
            y=30,
            line_dash="dot",
            line_color="green",
            annotation_text="Oversold",
            row=row,
            col=1,
        )

        # Set fixed range for RSI
        fig.update_yaxes(range=[0, 100], row=row, col=1)
    else:
        st.warning("RSI data not found. Skipping RSI plot.")


def add_macd_to_subplot(fig: go.Figure, data: pd.DataFrame, row: int = 3) -> None:
    """
    Add MACD indicator to a subplot.

    Args:
        fig: Plotly figure with subplots
        data: DataFrame containing MACD data
        row: Row number for the subplot
    """
    required_columns = ["MACD", "MACD_Signal", "MACD_Hist"]

    if all(col in data.columns for col in required_columns):
        # MACD line
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["MACD"],
                mode="lines",
                name="MACD Line",
                line=dict(color="blue", width=2),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # Signal line
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["MACD_Signal"],
                mode="lines",
                name="Signal Line",
                line=dict(color="orange", width=1),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # MACD Histogram as bar chart
        colors = ["red" if val < 0 else "green" for val in data["MACD_Hist"]]
        fig.add_trace(
            go.Bar(
                x=data["date"],
                y=data["MACD_Hist"],
                name="MACD Histogram",
                marker_color=colors,
                showlegend=False,
            ),
            row=row,
            col=1,
        )
    else:
        st.warning("MACD data not found. Skipping MACD plot.")


def create_rsi_chart(data: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create a standalone RSI chart.

    Args:
        data: DataFrame containing RSI data

    Returns:
        Plotly figure with RSI chart or None if data missing
    """
    if "RSI" not in data.columns:
        st.warning("RSI data not found.")
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="purple"),
        )
    )

    # Add reference lines
    fig.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Neutral")

    fig.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        height=300,
    )

    return fig


def create_macd_chart(data: pd.DataFrame) -> Optional[go.Figure]:
    """
    Create a standalone MACD chart.

    Args:
        data: DataFrame containing MACD data

    Returns:
        Plotly figure with MACD chart or None if data missing
    """
    required_columns = ["MACD", "MACD_Signal", "MACD_Hist"]

    if not all(col in data.columns for col in required_columns):
        st.warning("MACD data not found.")
        return None

    fig = go.Figure()

    # MACD line
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color="blue", width=2),
        )
    )

    # Signal line
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data["MACD_Signal"],
            mode="lines",
            name="Signal",
            line=dict(color="orange", width=1),
        )
    )

    # Histogram
    colors = ["red" if val < 0 else "green" for val in data["MACD_Hist"]]
    fig.add_trace(
        go.Bar(
            x=data["date"],
            y=data["MACD_Hist"],
            name="Histogram",
            marker_color=colors,
        )
    )

    # Add zero line
    fig.add_hline(
        y=0, line_dash="dash", line_color="black", annotation_text="Zero Line"
    )

    fig.update_layout(
        title="Moving Average Convergence Divergence (MACD)",
        xaxis_title="Date",
        yaxis_title="MACD",
        height=300,
    )

    return fig


def get_indicator_signals(data: pd.DataFrame) -> dict:
    """
    Generate current signals from technical indicators.

    Args:
        data: DataFrame with indicator data

    Returns:
        Dictionary with current indicator signals
    """
    if data.empty:
        return {}

    latest = data.iloc[-1]
    signals = {}

    # RSI signals
    if "RSI" in data.columns and pd.notna(latest["RSI"]):
        rsi_value = latest["RSI"]
        if rsi_value > 70:
            signals["RSI"] = {
                "value": rsi_value,
                "signal": "Overbought",
                "recommendation": "Sell",
            }
        elif rsi_value < 30:
            signals["RSI"] = {
                "value": rsi_value,
                "signal": "Oversold",
                "recommendation": "Buy",
            }
        else:
            signals["RSI"] = {
                "value": rsi_value,
                "signal": "Neutral",
                "recommendation": "Hold",
            }

    # MACD signals
    macd_cols = ["MACD", "MACD_Signal"]
    if all(col in data.columns for col in macd_cols):
        macd_value = latest["MACD"]
        signal_value = latest["MACD_Signal"]

        if pd.notna(macd_value) and pd.notna(signal_value):
            if macd_value > signal_value:
                signals["MACD"] = {"signal": "Bullish", "recommendation": "Buy"}
            elif macd_value < signal_value:
                signals["MACD"] = {"signal": "Bearish", "recommendation": "Sell"}
            else:
                signals["MACD"] = {"signal": "Neutral", "recommendation": "Hold"}

    return signals
