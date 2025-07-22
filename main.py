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
import itertools

# Import functions from our new modules from the 'app_modules' package
from app_modules.data_handler import (
    process_technical_indicators,
)
from app_modules.model_trainer import (
    dynamic_winsorize,
    model_drafts,
    tune_and_train_final_model,
)
from app_modules.plotter import plot_forecast
from prophet.diagnostics import (
    cross_validation,
)
from app_modules.config import load_environment_variables, EXCHANGE_CODE

# Import new UI modules - UPDATED IMPORTS
from app_modules.ui_intro import display_welcome_expander, display_stock_selection
from app_modules.forecast_summary import display_forecast_summary
from app_modules.performance_metrics import (
    display_accuracy_metrics,
    display_model_performance,
)
from app_modules.business_kpis import display_business_kpis
from app_modules.ui_appendix import display_appendix

# NEW IMPORT: Import the market_correlation module
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
) = display_stock_selection(FINNHUB_API_KEY, EXCHANGE_CODE, ts_av)

# --- Process Technical Indicators ---
price_col = "Adjusted Close"  # Define price_col here since it's used in data processing
data = process_technical_indicators(data, price_col)


## Forecasting

# Apply dynamic winsorization to raw data
data = dynamic_winsorize(data.copy(), price_col, percentiles=percentiles)

# Get training data
df_train = data[["Date", price_col, "winsorized"]].copy()
df_train = df_train.rename(columns={"Date": "ds"})

if train_period <= len(df_train):
    df_train = df_train[-train_period:]
else:
    # Silently adjust training period to available data length
    df_train = df_train


# Lambda function for cross validation metrics
def run_cross_validation(model_name):
    try:
        # Balanced CV: 2-4 folds maximum for performance
        available_data = len(df_train)
        
        # Use training period as initial, but ensure we have enough data for multiple folds
        cv_initial = min(train_period, int(available_data * 0.5))  # Max 50% of data for initial
        
        # Use period that gives us 2-3 folds maximum
        cv_period = max(period_unit, int((available_data - cv_initial) / 3))  # Divide remaining data into ~3 folds
        
        # Horizon should be reasonable for stock prediction (7-30 days)
        cv_horizon = min(forecast_period, 30, int(available_data * 0.1))  # Max 30 days or 10% of data
        
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


# Get metrics for baseline & winsorized models
scores_df = pd.DataFrame(columns=["mse", "rmse", "mae", "smape"])

# Check if user has modified parameters from defaults (slider position 5 = automated)
user_modified_params = (trend_flexibility != 5) or (seasonality_strength != 5)

if user_modified_params:
    optimization_mode = "Using your custom parameters. Skipping automated optimization."
    all_params = [
        {
            "changepoint_prior_scale": changepoint_prior,
            "seasonality_prior_scale": seasonality_prior,
        }
    ]
else:
    optimization_mode = "Running automated parameter optimization."
    param_grid = {
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
    }
    all_params = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

m, scores_df, forecast, best_params_dict, forecast_summary = (
    None,
    scores_df,
    pd.DataFrame(),
    {},
    "",
)

# Create a placeholder for progressive status updates
status_placeholder = st.empty()

# Step 1: Train baseline and winsorized models
status_placeholder.info(f"Step 1/3: Training baseline models... ({optimization_mode})")
if not df_train.empty and len(df_train) > 0:
    scores_df = model_drafts(
        df_train, scores_df, price_col, _cv_func=run_cross_validation
    )
    
    # Step 2: Compare models and select best data preparation approach
    status_placeholder.info("Step 2/3: Comparing baseline vs winsorized models...")
    if len(scores_df) >= 2:
        if scores_df.iloc[0]["rmse"] < scores_df.iloc[1]["rmse"]:
            df_train = df_train.rename(columns={price_col: "y"})
            chosen_approach = "raw data"
        else:
            df_train = df_train.rename(columns={"winsorized": "y"})
            chosen_approach = "winsorized data"
    else:
        st.warning(
            "Not enough model drafts for comparison. Using raw 'Adjusted Close' as target for final model."
        )
        df_train = df_train.rename(columns={price_col: "y"})
        chosen_approach = "raw data (fallback)"
    
    # Step 3: Train final model with hyperparameter tuning
    status_placeholder.info(f"Step 3/3: Training final model using {chosen_approach}...")
    
    m, scores_df, forecast, best_params_dict, forecast_summary = (
        tune_and_train_final_model(
            df_train,
            all_params,
            forecast_period,
            scores_df,
            _cv_func=run_cross_validation,
            summary_n_days_out=forecast_period,
        )
    )
    
    if m is not None and not forecast.empty:
        status_placeholder.success(f"All models trained successfully! Final model uses {chosen_approach}.")
    else:
        status_placeholder.error("Error: Model object or forecast is empty")
        st.stop()
else:
    status_placeholder.error("Error: Training data is empty")
    st.stop()

# Merge entire forecast w actual data & indicators
if not forecast.empty and not data.empty:
    # Limit historical data to match training period for visual consistency
    # Use the minimum of training period and available data
    actual_display_days = min(train_period, len(data))
    display_data = data[-actual_display_days:]
    
    forecast_df = pd.merge(
        left=display_data, right=forecast, right_on="ds", left_on="Date", how="right"
    )[
        [
            "ds",
            "Adjusted Close",
            "yhat",
            "yhat_lower",
            "yhat_upper",
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
            "Volume",
        ]
    ]
    forecast_df.rename(columns={"ds": "Date"}, inplace=True)
else:
    st.stop()  # Stop execution if forecast merge fails

# Get metrics
if len(scores_df) >= 3:
    scores_df.index = ["Baseline Model", "Winsorized Model", "Final Model"]
    scores_df = scores_df.reindex(sorted(scores_df.columns), axis=1)

# NEW STEP: Calculate Market Correlation
market_correlation = None
if not data.empty:
    with st.spinner("Calculating market correlation..."):
        # Pass the _ts_av object (which is the TimeSeries client) and the stock's data
        market_correlation = calculate_market_correlation(ts_av, data.copy())


# --- Calling plot_forecast ---
plot_forecast(forecast_df, ticker_name, selected_stock)

# --- UI: Display Forecast Summary Statements ---
display_forecast_summary(forecast_df, data, selected_stock)

# --- UI: Accuracy Metrics ---
display_accuracy_metrics(scores_df)

# --- UI: Business Intelligence KPIs ---
# UPDATED CALL: Pass market_correlation to display_business_kpis
display_business_kpis(forecast_df, data, stats, volatility, market_correlation)

# --- UI: Model Iterations and Performance ---
display_model_performance(scores_df, best_params_dict, selected_stock)

# --- UI: Appendix ---
display_appendix(tickers_data, m, forecast, data)
