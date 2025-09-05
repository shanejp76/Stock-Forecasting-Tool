"""
Data Pipeline Module

This module handles data preparation, cross-validation setup, and training data processing
for the stock forecasting application.

Author: Shane
Created: 2025-07-25
"""

import pandas as pd
import streamlit as st
from prophet.diagnostics import cross_validation
from .model_trainer import dynamic_winsorize


def prepare_training_data(data, percentiles, train_period):
    """
    Prepare and process training data with winsorization.

    Args:
        data: DataFrame with stock data
        percentiles: Tuple of percentiles for winsorization
        train_period: Number of days for training

    Returns:
        DataFrame: Prepared training data
    """
    # Apply dynamic winsorization to raw data
    processed_data = dynamic_winsorize(data.copy(), "close", percentiles=percentiles)

    # Reset index to get date as a column (BigQuery data has date as index)
    processed_data = processed_data.reset_index()

    # Get training data - handle both "Date" and "date" column names
    date_col = "Date" if "Date" in processed_data.columns else "date"
    df_train = processed_data[[date_col, "close", "winsorized"]].copy()
    df_train = df_train.rename(columns={date_col: "ds"})

    if train_period <= len(df_train):
        df_train = df_train[-train_period:]
    else:
        # Silently adjust training period to available data length
        df_train = df_train

    return df_train


def setup_cross_validation_function(
    df_train, train_period, period_unit, forecast_period
):
    """
    Setup cross-validation function with adaptive parameters based on data size.

    Args:
        df_train: Training DataFrame
        train_period: Training period in days
        period_unit: Period unit for CV
        forecast_period: Forecast period in days

    Returns:
        function: Cross-validation function
    """

    def run_cross_validation(model_name):
        try:
            # Balanced CV: 2-4 folds maximum for performance
            available_data = len(df_train)

            # Use training period as initial, but ensure we have enough data for multiple folds
            cv_initial = min(
                train_period, int(available_data * 0.5)
            )  # Max 50% of data for initial

            # Adaptive CV folds based on data size
            remaining_data = available_data - cv_initial

            # Determine target folds based on data size
            if available_data <= 250:  # ~1 year or less
                target_folds = 2
            elif available_data <= 500:  # ~2 years
                target_folds = 3
            elif available_data <= 1000:  # ~4 years
                target_folds = 4
            else:  # > 4 years
                target_folds = 5

            # Calculate period to achieve target folds, but respect limits
            target_period = remaining_data // target_folds
            min_period = 30  # Minimum 1 month spacing
            max_period = min(
                period_unit, remaining_data // 2
            )  # Cap by period_unit and ensure at least 2 folds

            cv_period = max(min_period, min(target_period, max_period))

            # Horizon should be reasonable for stock prediction (7-30 days)
            cv_horizon = min(
                forecast_period, 30, int(available_data * 0.1)
            )  # Max 30 days or 10% of data

            return cross_validation(
                model_name,
                initial=f"{cv_initial} days",
                period=f"{cv_period} days",
                horizon=f"{cv_horizon} days",
            )
        except ValueError as e:
            st.warning(
                f"Cross-validation failed: {e}. This might happen if your data is too short for the chosen initial, period, or horizon."
            )
            return pd.DataFrame()

    return run_cross_validation


def merge_forecast_with_data(forecast, data, train_period):
    """
    Merge forecast data with actual data and technical indicators.

    Args:
        forecast: DataFrame with forecast data
        data: DataFrame with actual stock data
        train_period: Training period for display consistency

    Returns:
        DataFrame: Merged forecast and actual data
    """
    if forecast.empty or data.empty:
        st.error("Cannot merge empty forecast or data")
        st.stop()

    # Limit historical data to match training period for visual consistency
    actual_display_days = min(train_period, len(data))
    display_data = data[-actual_display_days:].copy()

    # Ensure date is in a column for merge
    if "Date" not in display_data.columns:
        display_data = display_data.reset_index()
        # Handle both 'Date' and 'date' column names
        if "date" in display_data.columns and "Date" not in display_data.columns:
            display_data = display_data.rename(columns={"date": "Date"})

    forecast_df = pd.merge(
        left=display_data, right=forecast, right_on="ds", left_on="Date", how="right"
    )

    # Select only columns that actually exist in the merged DataFrame
    available_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]

    # Add optional columns if they exist (keep snake_case for internal processing)
    optional_cols = [
        "close",  # Primary price column
        "adjusted_close",  # Secondary price column
        "SMA50",
        "bb_upper",
        "bb_lower",
        "SMA20",
        "SMA100",
        "SMA200",
        "GoldenCross_Signal",
        "DeathCross_Signal",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "volume",  # Use snake_case internally
    ]

    for col in optional_cols:
        if col in forecast_df.columns:
            available_cols.append(col)

    forecast_df = forecast_df[available_cols]

    # Convert ds to lowercase date column for consistent chart usage (only if ds exists)
    if "ds" in forecast_df.columns:
        forecast_df.rename(columns={"ds": "date"}, inplace=True)
        # Also create uppercase Date for backward compatibility
        forecast_df["Date"] = forecast_df["date"]
    return forecast_df
