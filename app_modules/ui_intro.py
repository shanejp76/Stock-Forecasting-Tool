"""
UI Intro Module for Swing Ticker

This module provides Streamlit UI components for the introductory section of the app,
including the welcome message and stock selection interface. It handles initial data
loading, ticker lookup, and displays key statistics for the selected stock.

Functions:
    display_welcome_expander(): Displays the welcome message in an expander.
    display_stock_selection(FINNHUB_API_KEY, EXCHANGE_CODE, ts_av):
        Handles stock selection UI and loads initial data and statistics.

Author: Shane
Created: 2024-12-04
"""

# app_module/ui_intro.py
import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import date, timedelta

# Import functions from our other modules for data handling
from app_modules.data_handler import (
    load_finnhub_tickers,
    load_alpha_vantage_data,
    load_stock_data_hybrid,
    process_stock_data,
    calculate_stock_stats,
    determine_periods,
)


def display_welcome_expander():
    """Displays the welcome message in an expander."""
    st.title("Stock Forecasting Tool")
    with st.expander("Welcome! Click here to expand"):
        st.write(
            """
            -- Welcome to the Prophet Forecasting App: A Machine Learning & Analytics Exploration --

            **by Shane Peterson**

            This tool demonstrates the application of the Prophet forecasting model for predicting short-term price movements in stocks. This project focuses on showcasing the following machine learning and analytical skills:

            * **Data Acquisition & Preprocessing:** Fetching and cleaning financial data from reliable sources.
            * **User-Defined Parameters:** Allowing customizable model parameters for flexible analysis and validation of different forecasting approaches.
            * **Machine Learning Model Implementation:** Building and deploying Prophet forecasting models. By default, models are meticulously fine-tuned using advanced machine learning techniques and cross-validation to optimize for actionable business outcomes. When custom parameters are specified, the model uses those user-defined settings instead of auto-tuning.
            * **Time Series Forecasting:** Utilizing the Prophet model to predict stock prices.
            * **Hyperparameter Tuning:** Optimizing model performance through hyperparameter tuning. (see About section below)
            * **Model Evaluation:** Assessing model performance using appropriate metrics (e.g., SMAPE, RMSE). (see Model Iterations and Performance section below)

            The app includes comprehensive visualizations (candlestick charts, technical indicators including RSI, MACD, Bollinger Bands, and moving averages) alongside business intelligence KPIs that provide actionable insights for investment decision-making.

            **Note on Data:** *This app uses Alpha Vantage data with the following limitations: maximum 2-year historical period, daily market data only (no pre/post-market), potential API rate limits, and possible gaps in historical data for certain tickers. The model uses the most recent complete trading days available within this timeframe. Refer to the 'About' section for details on model validation methodology.*

            **Disclaimer: This app is for educational and demonstrative purposes only. It is not a financial recommendation and should not be used for actual trading decisions.**

            For more technical details please refer to the About section and the Appendix.
            """
        )


def display_stock_selection(FINNHUB_API_KEY, EXCHANGE_CODE, ts_av):
    """
    Handles stock selection UI and data loading/initial processing.
    Returns: selected_stock, ticker_name, data, stats, percentiles, volatility,
             period_unit, forecast_period, train_period, tickers_data,
             trend_flexibility, seasonality_strength, changepoint_prior,
             seasonality_prior, use_bigquery
    """
    st.subheader("-- Choose a Stock --")
    with st.expander("Click here to expand"):
        # Add data source selection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_stock = st.text_input(
                "Enter Symbol (Ticker List in Appendix)", value="goog"
            ).upper()
        with col2:
            use_bigquery = st.checkbox(
                "Use BigQuery",
                value=True,
                help="Use BigQuery data warehouse (faster, historical) or Alpha Vantage API (real-time, rate limited)",
            )

        tickers, tickers_data = load_finnhub_tickers(FINNHUB_API_KEY, EXCHANGE_CODE)

        data_load_state = st.text("-- Loading Data... --")
        ticker_name = ""
        for item in tickers_data:
            if item.get("symbol") == selected_stock:
                ticker_name = item.get("description", selected_stock)
                break

        data = pd.DataFrame()  # Initialize data to an empty DataFrame
        if selected_stock in tickers:
            # Use hybrid data loading approach
            data, data_source = load_stock_data_hybrid(
                selected_stock,
                date.today() - timedelta(days=2 * 365),  # 2 years back
                use_bigquery,
                ts_av,
            )
            if not data.empty:
                # Data loaded successfully
                data_load_state.empty()  # Clear the loading message
            else:
                data_load_state.text(
                    f"-- Failed to load data for '{selected_stock}'. Please try again or check API limits. --"
                )
                st.stop()
        else:
            data_load_state.text(
                f"-- '{selected_stock}' is not a valid Symbol. Please refresh and enter a symbol from the Ticker List in the Appendix below. --"
            )
            st.stop()

        # Process and filter data
        num_years_back = 2
        TODAY = date.today()
        DYNAMIC_START_DATE = TODAY - timedelta(days=num_years_back * 365)
        data = process_stock_data(data, DYNAMIC_START_DATE)

        if data.empty:
            st.error(
                f"No data available for {selected_stock} after {DYNAMIC_START_DATE.strftime('%Y-%m-%d')}. "
                f"Please choose a different stock, adjust the date range in the code, or consider a premium API for more historical data."
            )
            st.stop()

        price_col = "Adjusted Close"
        stats, percentiles, volatility = calculate_stock_stats(
            data, selected_stock, price_col
        )

        # --- Displaying Stock Statistics in a single-line DataFrame ---
        if stats:
            stats_df_for_display = pd.DataFrame([stats])
            desired_column_order = [
                "Symbol",
                "Current Price",
                "Current Volume",
                "Daily % Change",
                "YTD % Change",
                "Annualized Volatility",
                "52-Week High",
                "52-Week Low",
                "Earliest Data Date",
                "Last Data Date",
            ]
            existing_ordered_columns = [
                col
                for col in desired_column_order
                if col in stats_df_for_display.columns
            ]
            stats_df_for_display = stats_df_for_display[existing_ordered_columns]

            # Rename 'Average Daily Percentage Change' to 'Daily % Change'
            if "Average Daily Percentage Change" in stats_df_for_display.columns:
                stats_df_for_display = stats_df_for_display.rename(
                    columns={"Average Daily Percentage Change": "Daily % Change"}
                )

            st.dataframe(stats_df_for_display, height=60, use_container_width=True)
        else:
            st.warning("Stock statistics are not available.")

        # --- Model Configuration ---
        st.markdown("**-- Model Configuration --**")

        # Training Duration Slider (moved to bottom of section)
        training_days = st.slider(
            "Training Data Duration (Trading Days)",
            min_value=30,
            max_value=500,  # ~2 years of trading days
            value=500,  # Default to maximum available
            step=25,
            help="Select the number of trading days (excluding weekends/holidays) for model training. ~250 trading days = 1 year, ~500 trading days = 2 years. More data may improve accuracy but increase processing time.",
        )

        # Parameter Controls (simplified sliders)
        trend_flexibility = st.slider(
            "Trend Flexibility",
            min_value=1,
            max_value=10,
            value=5,  # Maps to 0.1 (the default)
            help="How quickly the model adapts to trend changes. Lower = more conservative, Higher = more responsive to changes",
        )

        seasonality_strength = st.slider(
            "Seasonality Strength",
            min_value=1,
            max_value=10,
            value=5,  # Maps to 1.0 (the default)
            help="How much seasonal variation the model captures. Lower = smoother patterns, Higher = more pronounced seasonal effects",
        )

        # Convert simple 1-10 scale to log scale parameters
        # Changepoint: 1->0.001, 5->0.1, 10->0.5 (logarithmic mapping)
        changepoint_prior = np.exp(
            np.log(0.001) + (trend_flexibility - 1) * (np.log(0.5) - np.log(0.001)) / 9
        )

        # Seasonality: 1->0.01, 5->1.0, 10->10.0 (logarithmic mapping)
        seasonality_prior = np.exp(
            np.log(0.01)
            + (seasonality_strength - 1) * (np.log(10.0) - np.log(0.01)) / 9
        )

        # Determine training and forecast periods
        period_unit, forecast_period, train_period = determine_periods(
            data, volatility, training_days
        )

    return (
        selected_stock,
        ticker_name,
        data,
        stats,
        percentiles,
        volatility,
        period_unit,
        forecast_period,
        train_period,
        tickers_data,
        trend_flexibility,
        seasonality_strength,
        changepoint_prior,
        seasonality_prior,
        use_bigquery,
    )
