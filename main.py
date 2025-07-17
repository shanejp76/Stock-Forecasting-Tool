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
    load_finnhub_tickers,  # Still needed for initial API key check in main
    load_alpha_vantage_data,  # Still needed for initial API key check in main
    process_stock_data,  # Still needed for initial API key check in main
    calculate_stock_stats,  # Still needed for initial API key check in main
    process_technical_indicators,
    determine_periods,
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

# Import new UI modules
from app_modules.ui_intro import display_welcome_expander, display_stock_selection
from app_modules.ui_metrics import (
    display_forecast_summary,
    display_accuracy_metrics,
    display_business_kpis,
    display_model_performance,
)
from app_modules.ui_appendix import display_appendix


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
    st.warning(
        f"Training period ({train_period}) is larger than available data ({len(df_train)}). Adjusting training period to available data length."
    )
    df_train = df_train


# Lambda function for cross validation metrics
def run_cross_validation(model_name):
    try:
        return cross_validation(
            model_name,
            initial=f"{train_period} days",
            period=f"{period_unit} days",
            horizon=f"{forecast_period} days",
        )
    except ValueError as e:
        st.warning(
            f"Cross-validation failed: {e}. This might happen if your data is too short for the chosen initial, period, or horizon."
        )
        return pd.DataFrame()


# Get metrics for baseline & winsorized models
scores_df = pd.DataFrame(columns=["mse", "rmse", "mae", "smape"])

data_load_state = st.text(
    "-- Please wait while the Baseline & Winsorized models train... --"
)
if not df_train.empty and len(df_train) > 0:
    scores_df = model_drafts(
        df_train, scores_df, price_col, _cv_func=run_cross_validation
    )
    data_load_state.text("-- Baseline & Winsorized models Trained. --")
else:
    data_load_state.text("-- Error in training models: Training data is empty. --")
    st.stop()

if len(scores_df) >= 2:
    if scores_df.iloc[0]["rmse"] < scores_df.iloc[1]["rmse"]:
        df_train = df_train.rename(columns={price_col: "y"})
    else:
        df_train = df_train.rename(columns={"winsorized": "y"})
else:
    st.warning(
        "Not enough model drafts for comparison. Using raw 'Adjusted Close' as target for final model."
    )
    df_train = df_train.rename(columns={price_col: "y"})

param_grid = {
    "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
    "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
}
all_params = [
    dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
]

data_load_state = st.text("-- Please wait while the Final Model trains... --")
m, scores_df, forecast, best_params_dict, forecast_summary = (
    None,
    scores_df,
    pd.DataFrame(),
    {},
    "",
)

if not df_train.empty and len(df_train) > 0:
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
        data_load_state.text("-- Final Model Trained. --")
    else:
        data_load_state.text(
            "-- Error in training final model: Model object or forecast is empty. --"
        )
        st.stop()
else:
    data_load_state.text("-- Error in training final model: Training data is empty. --")
    st.stop()

# Merge entire forecast w actual data & indicators
if not forecast.empty and not data.empty:
    forecast_df = pd.merge(
        left=data, right=forecast, right_on="ds", left_on="Date", how="right"
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
    st.error("Cannot merge forecast: forecast or data is empty.")
    st.stop()

# Get metrics
if len(scores_df) >= 3:
    scores_df.index = ["Baseline Model", "Winsorized Model", "Final Model"]
    scores_df = scores_df.reindex(sorted(scores_df.columns), axis=1)
else:
    st.warning("Not enough model scores for comparison. Displaying available scores.")

# --- Calling plot_forecast ---
plot_forecast(forecast_df, ticker_name, selected_stock)

# --- UI: Display Forecast Summary Statements ---
display_forecast_summary(forecast_df, data, selected_stock)

# --- UI: Accuracy Metrics ---
display_accuracy_metrics(scores_df)

# --- UI: Business Intelligence KPIs ---
display_business_kpis(forecast_df, data, volatility)

# --- UI: Model Iterations and Performance ---
display_model_performance(scores_df, best_params_dict, selected_stock)

# --- UI: Appendix ---
display_appendix(tickers_data, m, forecast, data)
