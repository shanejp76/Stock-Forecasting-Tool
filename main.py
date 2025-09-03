"""
Swing Ticker Streamlit App

This module serves as the main entry point for the Swing Ticker application.
It loads API keys, fetches and processes stock data, applies forecasting models,
calculates technical indicators and market correlation, and displays results
and business KPIs through an interactive Streamlit UI.

Modules imported:
- Data handling and technical indicators
- Model training and evaluation
- Forecast plotting
- UI components and business KPIs
- Market correlation analysis

Author: Shane
Created: 2024-12-04
"""

# main.py
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import date, timedelta
from alpha_vantage.timeseries import TimeSeries

# Import functions from our new modules from the 'app_modules' package
from app_modules.data_handler import process_technical_indicators
from app_modules.plotter import plot_forecast
from app_modules.config import load_environment_variables, EXCHANGE_CODE

# Import new pipeline modules
from app_modules.data_pipeline import (
    prepare_training_data,
    setup_cross_validation_function,
    merge_forecast_with_data,
)
from app_modules.model_orchestrator import (
    determine_optimization_strategy,
    execute_training_pipeline,
    format_scores_dataframe,
)

# Import UI modules
from app_modules.ui_intro import display_welcome_expander, display_stock_selection
from app_modules.forecast_summary import display_forecast_summary
from app_modules.performance_metrics import (
    display_accuracy_metrics,
    display_model_performance,
)
from app_modules.business_kpis import display_business_kpis
from app_modules.ui_appendix import display_appendix
from app_modules.market_correlation import calculate_market_correlation


# Load API keys
ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY = load_environment_variables()

if ALPHA_VANTAGE_API_KEY is None:
    st.error(
        "ERROR: Alpha Vantage API key not found. Make sure it's in your .env file!"
    )
    st.stop()
else:
    ts_av = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")


# --- UI: Welcome Section ---
display_welcome_expander()

# --- UI & Data: Choose a Stock and Load Data ---
(
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
) = display_stock_selection(FINNHUB_API_KEY, EXCHANGE_CODE, ts_av)

# --- Process Technical Indicators ---
# Define price_col based on available columns - PRIORITIZE UNADJUSTED CLOSE for swing trading
if "Close" in data.columns:
    price_col = "Close"
elif "close" in data.columns:
    price_col = "close"
elif "adjusted_close" in data.columns:
    price_col = "adjusted_close"
else:
    price_col = "close"  # Default fallback to snake_case

st.info(f"Using price column: **{price_col}** for modeling and analysis")
data = process_technical_indicators(data, price_col)


## Forecasting

# --- Prepare Training Data ---
df_train = prepare_training_data(data, price_col, percentiles, train_period)

# --- Setup Cross-Validation Function ---
run_cross_validation = setup_cross_validation_function(
    df_train, train_period, period_unit, forecast_period
)

# --- Determine Optimization Strategy ---
optimization_mode, all_params, user_modified_params = determine_optimization_strategy(
    trend_flexibility, seasonality_strength, changepoint_prior, seasonality_prior
)

# --- Execute Model Training Pipeline ---
m, scores_df, forecast, best_params_dict, forecast_summary, chosen_approach = (
    execute_training_pipeline(
        df_train, price_col, all_params, forecast_period, run_cross_validation
    )
)

# --- Merge Forecast with Data ---
forecast_df = merge_forecast_with_data(forecast, data, train_period)

# --- Format Scores DataFrame ---
scores_df = format_scores_dataframe(scores_df)

# NEW STEP: Calculate Market Correlation
market_correlation = None
if not data.empty:
    with st.spinner("Calculating market correlation..."):
        # Pass the _ts_av object, stock data, and use_bigquery flag
        market_correlation = calculate_market_correlation(
            ts_av, data.copy(), use_bigquery
        )


# --- Calling plot_forecast ---
plot_forecast(forecast_df, ticker_name, selected_stock)

# --- UI: Display Forecast Summary Statements ---
display_forecast_summary(forecast_df, data, selected_stock, scores_df, price_col)

# --- UI: Accuracy Metrics ---
display_accuracy_metrics(scores_df)

# Add dividing line after model accuracy section
st.markdown("---")

# --- UI: Business Intelligence KPIs ---
# UPDATED CALL: Pass market_correlation to display_business_kpis
display_business_kpis(
    forecast_df, data, stats, volatility, market_correlation, price_col
)

# --- UI: Model Iterations and Performance ---
display_model_performance(
    scores_df, best_params_dict, selected_stock, user_modified_params
)

# --- UI: Appendix ---
display_appendix(tickers_data, m, forecast, data)
