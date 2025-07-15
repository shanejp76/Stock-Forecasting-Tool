import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import date, timedelta
import requests
from alpha_vantage.timeseries import TimeSeries

# Import functions from our new modules from the 'app_modules' package
from app_modules.data_handler import (
    load_finnhub_tickers,
    load_alpha_vantage_data,
    process_stock_data,
    calculate_stock_stats,
    process_technical_indicators,
    determine_periods,
)
from app_modules.model_trainer import (
    dynamic_winsorize,
    model_drafts,
    tune_and_train_final_model,
)
from app_modules.plotter import plot_forecast
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from prophet import Prophet
import itertools
import numpy as np

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
FINNHUB_API_KEY = os.getenv("FINNHub_API_KEY", "YOUR_FINNHUB_API_KEY")
EXCHANGE_CODE = "US"

if ALPHA_VANTAGE_API_KEY is None:
    st.error(
        "ERROR: Alpha Vantage API key not found. Make sure it's in your .env file!"
    )
    st.stop()
else:
    ts_av = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")


st.title("Stock Forecasting Tool")

with st.expander("Welcome! Click here to expand"):
    st.write(
        """
    -- Welcome to the Prophet Forecasting App: A Data Science Exploration --

    **by Shane Peterson**

    This tool demonstrates the application of the Prophet forecasting model for predicting short-term price movements in stocks. This project focuses on showcasing the following data science skills:

    * **Time Series Forecasting:** Utilizing the Prophet model to predict stock prices.
    * **Hyperparameter Tuning:** Optimizing model performance through hyperparameter tuning. (see About section below)
    * **Model Evaluation:** Assessing model performance using appropriate metrics (e.g., SMAPE, RMSE). (see Accuracy Metrics section below)

    The app includes visualizations such as line charts, moving averages, and Bollinger Bands to provide context for the forecast.

    **Note on Data:** *This app uses a limited historical data source for demonstration purposes. Refer to the 'About' section for details on model validation and data limitations.*

    **Disclaimer: This app is for educational and demonstrative purposes only. It is not a financial recommendation and should not be used for actual trading decisions.**

    For more technical details please refer to the About section and the Appendix.
    """
    )


## Choose a Stock
st.subheader("-- Choose a Stock --")
with st.expander("Click here to expand"):
    selected_stock = st.text_input(
        "Enter Symbol (Ticker List in Appendix)", value="goog"
    ).upper()

    tickers, tickers_data = load_finnhub_tickers(FINNHUB_API_KEY, EXCHANGE_CODE)

    data_load_state = st.text("-- Loading Data... --")
    ticker_name = ""
    for item in tickers_data:
        if item.get("symbol") == selected_stock:
            ticker_name = item.get("description", selected_stock)
            break

    if selected_stock in tickers:
        # Pass ts_av with an underscore in main.py call as well
        data = load_alpha_vantage_data(_ts_av=ts_av, ticker=selected_stock)
        if not data.empty:
            data_load_state.text(f"-- {ticker_name} Data Loaded. --")
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

    if stats:
        stats_window_df = pd.DataFrame([stats])
        st.write(stats_window_df)
    else:
        st.warning("Stock statistics are not available.")

    # Determine training and forecast periods
    period_unit, forecast_period, train_period = determine_periods(data, volatility)

# --- Process Technical Indicators ---
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
# This needs to be defined here because `period_unit`, `train_period`, `forecast_period` are determined dynamically
cv_func = lambda model_name: cross_validation(
    model_name,
    initial=f"{train_period} days",
    period=f"{period_unit} days",
    horizon=f"{forecast_period} days",
)

# Get metrics for baseline & winsorized models
scores_df = pd.DataFrame(columns=["mse", "rmse", "mae", "smape"])

data_load_state = st.text(
    "-- Please wait while the Baseline & Winsorized models train... --"
)
if not df_train.empty and len(df_train) > 0:
    # Pass cv_func with an underscore
    scores_df = model_drafts(df_train, scores_df, price_col, _cv_func=cv_func)
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
if not df_train.empty and len(df_train) > 0:
    # Pass cv_func with an underscore
    m, scores_df, forecast, best_params_dict = tune_and_train_final_model(
        df_train, all_params, forecast_period, scores_df, _cv_func=cv_func
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
            "RSI",  # ADDED
            "MACD",  # ADDED
            "MACD_Signal",  # ADDED
            "MACD_Hist",  # ADDED
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
    st.warning(
        "Not enough model scores to label all iterations. Displaying available scores."
    )

# --- Calling plot_forecast ---
plot_forecast(forecast_df, ticker_name, selected_stock)


## Chart Tips
st.subheader("-- Chart Tips --")
with st.expander("Click here to expand"):
    st.write("* Use the slider (above) to select a date range")
    st.write("* Click items in the legend to show/hide indicators")
    st.write(
        "* Hover in the upper-right corner of graph to reveal controls. Go fullscreen and explore!"
    )


## Accuracy Metrics
st.subheader("**-- Predicted Accuracy --**")
if (
    len(scores_df) > 2
    and "smape" in scores_df.columns
    and "Final Model" in scores_df.index
):
    st.subheader(f'{100-(round(scores_df.loc["Final Model"]["smape"]*100, 2))}%')
else:
    st.write(
        "Accuracy metrics not fully available yet. (Requires successful training of all 3 models)"
    )

st.subheader("-- Model Iterations --")
with st.expander("Click here to expand"):
    st.write(
        "The tables below illustrate the methodical and iterative approach to model refinement and performance optimization, a critical skill in data science. We begin with a Baseline Model using raw data, then introduce a Winsorized Model to address data challenges posed by outliers, demonstrating the implementation of effective data preprocessing solutions. Finally, the Final Model showcases performance optimization through rigorous hyperparameter tuning. Each stage is rigorously evaluated using cross-validation and a comparison of performance metrics (SMAPE, RMSE, MAE, MSE), clearly demonstrating the tangible improvements gained at each step. This process highlights a comprehensive data science workflow, from identifying data-driven problems and implementing solutions to optimizing model performance and validating improvements for enhanced predictive accuracy and business value."
    )

    if "Baseline Model" in scores_df.index:
        st.write("-- Baseline Model --")
        st.dataframe(scores_df.loc[["Baseline Model"]], width=500)
    if "Winsorized Model" in scores_df.index:
        st.write("-- Winsorized Model --")
        st.dataframe(scores_df.loc[["Winsorized Model"]], width=500)
    if "Final Model" in scores_df.index:
        st.write("-- Final Model --")
        st.dataframe(scores_df.loc[["Final Model"]], width=500)
    else:
        st.write("Not all model iteration metrics are available.")

    st.write(
        "In the context of time series forecasting, 'error' refers to the difference between the actual value of a variable at a specific point in time and the value predicted by a forecasting model. In this case, the metrics will specifically measure the error between the stock's closing price and the forecast trained on the closing price."
    )

    if (
        len(scores_df) > 2
        and "mae" in scores_df.columns
        and "smape" in scores_df.columns
        and "rmse" in scores_df.columns
        and "Final Model" in scores_df.index
    ):
        st.write(
            f"* Mean Absolute Error (MAE) - a MAE of {round(scores_df.loc['Final Model']['mae'], 4)} implies that, on average, the model's predictions are off by approximately ${round(scores_df.loc['Final Model']['mae'], 2)}."
        )
        st.write(
            f"* Symmetric Mean Absolute Percentage Error (SMAPE) - a SMAPE of {round(scores_df.loc['Final Model']['smape'], 4)} means that, on average, the model's predictions are {round(scores_df.loc['Final Model']['smape'] * 100, 2)}% off from the actual values."
        )
        st.write(
            "* Mean Squared Error (MSE) - this squares the errors, giving more weight to larger errors. A lower MSE indicates better accuracy."
        )
        st.write(
            f"* Root Mean Squared Error (RMSE) -  The square root of MSE. It is in the same units as the original data, making it easier to interpret. The RMSE of {round(scores_df.loc['Final Model']['rmse'], 4)} suggests that the model's predictions can deviate from the actual values by up to ${round(scores_df.loc['Final Model']['rmse'], 2)} in some cases."
        )
    else:
        st.write(
            "Detailed metric descriptions are not available due to incomplete model training."
        )

    st.write(
        "Beyond statistical measures, these metrics translate directly into business impact. For instance, a lower Mean Absolute Error (MAE) signifies that, on average, our forecasts are closer to the actual stock price. This precision is critical as it directly reduces potential financial risk by providing more accurate price expectations. Similarly, the Root Mean Squared Error (RMSE) quantifies the typical magnitude of our prediction errors, which is invaluable for informing robust risk assessments and setting realistic expectations for portfolio management. The Symmetric Mean Absolute Percentage Error (SMAPE) further enhances this by providing a clear, percentage-based understanding of forecast accuracy, allowing for straightforward interpretation of how far off our predictions are, on average, from the actual values."
    )

## About
st.subheader("-- About --")
with st.expander("Click here to expand"):
    about_str = f"""
    **-- Purpose & Business Value --**

    As a passionate trader, I developed this application to streamline my decision-making process. It leverages fundamental data science concepts, including data engineering and analytics, to provide actionable insights.

    The app features a user-friendly interface with a line chart, Bollinger Bands, and a Simple Moving Average (SMA) for visual analysis of price trends.

    **-- The Model --**

    To enhance trading strategies and support robust decision-making, this application integrates a Prophet forecasting model, meticulously fine-tuned using advanced machine learning techniques. These methods were chosen not just for technical accuracy, but specifically to **optimize for actionable business outcomes.**

    Key Model Enhancements chosen for their impact on decision-making:
    * **Winsorization:** This technique was applied to improve the model's robustness against extreme price fluctuations (outliers). By mitigating the impact of unusual data points, the model generates **more stable and reliable predictions, reducing noise and leading to more confident trading decisions.** The thresholds are dynamically adjusted based on the stock's volatility to ensure relevance.
    * **Adaptive Training Data:** The size of the training dataset is dynamically adjusted based on the stock's volatility and available data. This ensures the model is trained on the most relevant historical period, which is crucial for **maintaining forecast agility and relevance in fluctuating market conditions.**
    * **Hyperparameter Tuning:** Through a cross-validated grid search, key model parameters (changepoint_prior_scale and seasonality_prior_scale) are systematically optimized. This process ensures the model learns the underlying patterns most effectively, leading to **highly accurate forecasts that directly translate into improved decision quality and reduced financial risk.**

    **Data Preparation:** Before modeling, careful data preparation steps were undertaken. While the raw data from our source (Alpaca) is robust and guarantees cleanliness, I performed essential transformations such as date alignment, feature engineering (e.g., adding technical indicators), and dynamic winsorization to prepare the data optimally for the Prophet model.

    By combining these refinements with a cross-validated grid search, this application provides a robust forecasting tool.

    """
    if best_params_dict:
        about_str += f"For '{selected_stock}', optimal values are: changepoint_prior_scale: {best_params_dict['changepoint_prior_scale']:.3f}, seasonality_prior_scale: {best_params_dict['seasonality_prior_scale']:.3f}.\n\n"
    else:
        about_str += "Optimal hyperparameters could not be determined.\n\n"

    about_str += """
    **Cross-validation is paramount to ensuring the model's generalizability and reliability**, directly translating to **trustworthiness in business insights**. By rigorously evaluating the model's performance on multiple, unseen subsets of the data during the grid search, we can select hyperparameters that are not overfitted to a specific dataset. This robust validation process ensures that the model performs consistently on new data, providing a dependable foundation for trading decisions and strategic planning. Check out Model Iterations in the More Metrics section (above) to observe the model's improvement over its learning cycles.

    **-- Data Source & Considerations --**

    The core forecasting methodology behind this application was rigorously tested and validated in a prior experiment involving 150 stocks of varying volatility, utilizing comprehensive historical data from `yfinance`. That experiment demonstrated a Symmetric Mean Absolute Percentage Error (SMAPE) of approximately 15% across the diverse dataset, showcasing the model's general effectiveness.

    For this live demonstration, market data is sourced via a free-tier API (Alpha Vantage). Due to API limitations, the available historical data for analysis and forecasting is significantly shortened compared to the original validation experiment. While this allows for a functional demonstration of the forecasting capabilities, **a diligent business intelligence analyst would note that this limited data history can impact the robustness and long-term reliability of the forecasts presented here.** The purpose remains to illustrate the application's functionality and demonstrate a comprehensive analytical workflow, rather than to provide definitive predictive accuracy based on constrained data.

    **-- Trading Tips --**

    By combining the forecasting model with visual aids like a line chart, Bollinger Bands, and SMAs, I'm able to identify potential entry and exit points with greater confidence, ultimately refining my trading decisions.

    By selecting a stock ticker, the app displays important background information like historical highs/lows, percentage change, volatility, and current price alongside the chart. This comprehensive tool empowers more informed trading decisions and refined trading strategies.
    """
    st.write(about_str)

## Appendix
st.subheader("-- Appendix --")
with st.expander("Click here to expand"):
    st.subheader("-- Ticker List --")
    if not pd.DataFrame(tickers_data).empty:
        st.write(pd.DataFrame(tickers_data))
    else:
        st.write("Ticker list not available.")

    st.subheader("-- Forecast Components --")
    if m is not None and not forecast.empty:
        try:
            st.write(
                "The 'Forecast Components' provide invaluable business intelligence by decomposing the overall forecast into its fundamental drivers. These components, including trend, weekly seasonality, and yearly seasonality, reveal the underlying patterns that Prophet has identified in the historical data. (Note: Yearly seasonality may be unavailable if given limited historical data.) For a business analyst, this isn't just about seeing what the forecast is, but crucially, understanding why the model predicts what it does. By visualizing these individual contributions, analysts can gain insights into long-term growth or decline, regular weekly fluctuations, and recurring annual cycles, empowering them to make more informed decisions based on a clear understanding of the forces shaping future outcomes."
            )
            fig2 = m.plot_components(forecast)
            st.write(fig2)
        except Exception as e:
            st.error(f"Error plotting forecast components: {e}")
    else:
        st.write("Forecast components not available: model or forecast is empty.")

    st.subheader("-- Forecast Grid --")
    if not forecast.empty:
        st.write(forecast)
    else:
        st.write("Forecast grid not available: forecast is empty.")

    st.subheader("-- Raw Data (Filtered) --")
    if not data.empty:
        st.write(data)
    else:
        st.write("Raw data not available: data is empty.")
