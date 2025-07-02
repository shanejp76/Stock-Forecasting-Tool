import requests
import streamlit as st
import streamlit.components.v1 as components
from datetime import date, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import ta
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import itertools
import io
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_KEY")

if ALPHA_VANTAGE_API_KEY is None:
    st.error(
        "ERROR: Alpha Vantage API key not found. Make sure it's in your .env file!"
    )
    st.stop()
else:
    ts_av = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")

st.title("Stock Forecasting Tool")

with st.expander("-- Welcome! Click here to expand --"):
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

### Ticker Selection Searchbar
st.subheader("-- Choose a Stock --")
with st.expander("-- Click here to expand --"):
    selected_stock = st.text_input(
        "Enter Symbol (Ticker List in Appendix)", value="goog"
    ).upper()

    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
    EXCHANGE_CODE = "US"

    tickers = []
    tickers_data = []
    if FINNHUB_API_KEY != "YOUR_FINNHUB_API_KEY" and FINNHUB_API_KEY != "":
        url = f"https://finnhub.io/api/v1/stock/symbol?exchange={EXCHANGE_CODE}&token={FINNHUB_API_KEY}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            tickers_data = response.json()
            tickers = [item["symbol"] for item in tickers_data]
        except requests.exceptions.RequestException as e:
            st.warning(
                f"-- Error fetching Finnhub data: {e}. Ticker search functionality might be limited. --"
            )
            tickers = ["IBM", "GOOG", "MSFT", "AAPL"]
            tickers_data = [
                {"symbol": t, "description": f"{t} Company"} for t in tickers
            ]
    else:
        st.warning(
            "Finnhub API key not set or is placeholder. Ticker search will use default list."
        )
        tickers = ["IBM", "GOOG", "MSFT", "AAPL"]
        tickers_data = [{"symbol": t, "description": f"{t} Company"} for t in tickers]

    @st.cache_data
    def load_data(ticker):
        try:
            data, meta_data = ts_av.get_daily(ticker, outputsize="full")

            data.columns = ["Open", "High", "Low", "Close", "Volume"]
            data.index.name = "Date"
            data.reset_index(inplace=True)
            data["Date"] = pd.to_datetime(data["Date"]).dt.date
            data["Adjusted Close"] = data["Close"]
            return data
        except Exception as e:
            st.error(f"Error loading data for {ticker}: {e}")
            st.info(
                f"This often happens if you've hit API rate limits. Please try again tomorrow."
            )
            return pd.DataFrame()

    data_load_state = st.text("-- Loading Data... --")
    ticker_name = ""
    for item in tickers_data:
        if item.get("symbol") == selected_stock:
            ticker_name = item.get("description", selected_stock)
            break

    if selected_stock in tickers:
        data = load_data(selected_stock)
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

    data.Date = pd.to_datetime(data.Date)
    data.Date = data.Date.astype("datetime64[ns]")

    data = data[::-1].reset_index(drop=True)

    num_years_back = 2
    TODAY = date.today()
    DYNAMIC_START_DATE = TODAY - timedelta(days=num_years_back * 365)

    DYNAMIC_START_DATE = pd.to_datetime(DYNAMIC_START_DATE)

    data = data[data["Date"] >= DYNAMIC_START_DATE].reset_index(drop=True)

    if data.empty:
        st.error(
            f"No data available for {selected_stock} after {DYNAMIC_START_DATE.strftime('%Y-%m-%d')}. "
            f"Please choose a different stock, adjust the date range in the code, or consider a premium API for more historical data."
        )
        st.stop()

    price_col = "Adjusted Close"
    stats = {}

    if not data.empty and price_col in data.columns:
        stats["Symbol"] = selected_stock
        stats["Current Price"] = round(data[price_col].iloc[-1], 2)
        stats["Current Volume"] = data["Volume"].iloc[-1]
        data["daily_returns"] = data[price_col].pct_change()
        volatility = data["daily_returns"].std() * np.sqrt(252)

        percentiles = (0.1, 0.9)
        if volatility < 0.2:
            category = "Low"
            percentiles = (0.15, 0.85)
        elif volatility < 0.4:
            category = "Medium-Low"
            percentiles = (0.1, 0.9)
        elif volatility < 0.6:
            category = "Medium"
            percentiles = (0.1, 0.9)
        elif volatility < 0.8:
            category = "Medium-High"
            percentiles = (0.05, 0.95)
        else:
            category = "High"
            percentiles = (0.05, 0.95)
        stats["Annualized Volatility"] = category
        stats["Average Daily Percentage Change"] = (
            str(round(data["daily_returns"].mean() * 100, 4)) + " %"
        )
    else:
        st.warning(
            "Could not retrieve stock statistics. Please ensure the symbol is valid and data loaded correctly."
        )
        st.stop()

    if stats:
        stats_window_df = pd.DataFrame([stats])
        st.write(stats_window_df)
    else:
        st.warning("Stock statistics are not available.")

    # Get Stock Age & Set training & forecast periods
    if not data.empty:
        data_len_years = len(data) / 365
        if data_len_years < 2:
            period_unit = int(len(data) / 4)
            forecast_period = period_unit
            train_period = len(data)  # Train on all available data
            stock_age = "young"
        else:
            period_unit = 365
            forecast_period = period_unit
            train_period = (
                forecast_period * 4 if volatility < 0.6 else forecast_period * 8
            )
            stock_age = "seasoned"

        # Ensure these are always integers before being passed around
        period_unit = int(period_unit)
        forecast_period = int(forecast_period)
        train_period = int(train_period)

    else:
        st.warning("Data is empty, cannot determine stock age or set training periods.")
        st.stop()

# --- Process Technical Indicators ---
if not data.empty and price_col in data.columns:
    data["SMA50"] = data[price_col].rolling(window=50).mean()
    indicator_bb = ta.volatility.BollingerBands(
        close=data[price_col], window=20, window_dev=2
    )
    data["bb_upper"] = indicator_bb.bollinger_hband()
    data["bb_lower"] = indicator_bb.bollinger_lband()
else:
    st.error(
        "Cannot process indicators: 'Adjusted Close' column not found or data is empty."
    )
    st.stop()

# --- FORECASTING ---


# Winsorize Function
def dynamic_winsorize(df, column, window_size=30, percentiles=(0.05, 0.95)):
    df["rolling_lower"] = (
        df[column].rolling(window=window_size, min_periods=1).quantile(percentiles[0])
    )
    df["rolling_upper"] = (
        df[column].rolling(window=window_size, min_periods=1).quantile(percentiles[1])
    )
    df["winsorized"] = df[column]
    df.loc[df[column] < df["rolling_lower"], "winsorized"] = df["rolling_lower"]
    df.loc[df[column] > df["rolling_upper"], "winsorized"] = df["rolling_upper"]
    return df


# Apply dynamic winsorization to raw data
if "percentiles" not in locals():
    percentiles = (0.05, 0.95)
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
cv_func = lambda model_name: cross_validation(
    model_name,
    initial=f"{train_period} days",
    period=f"{period_unit} days",
    horizon=f"{forecast_period} days",
)

# Get metrics for baseline & winsorized models
scores_df = pd.DataFrame(columns=["mse", "rmse", "mae", "smape"])


@st.cache_resource
def model_drafts(df_train_input, scores_df_input):
    current_scores_df = scores_df_input.copy()
    for col_name in [price_col, "winsorized"]:
        if col_name not in df_train_input.columns:
            st.warning(
                f"Column '{col_name}' not found in training data. Skipping model draft for this column."
            )
            continue

        m = Prophet()
        df_train_renamed = df_train_input[["ds", col_name]].rename(
            columns={col_name: "y"}
        )
        try:
            m.fit(df_train_renamed)
            df_cv = cv_func(m)
            if not df_cv.empty:
                df_p = performance_metrics(df_cv, rolling_window=1)
                current_scores_df = pd.concat(
                    [current_scores_df, df_p[["mse", "rmse", "mae", "smape"]]],
                    ignore_index=True,
                )
            else:
                st.warning(
                    f"Cross-validation for {col_name} returned no results. Skipping metric calculation."
                )
        except Exception as e:
            st.error(f"Error training model draft for {col_name}: {e}")
            pass
    return current_scores_df


data_load_state = st.text(
    "-- Please wait while the Baseline & Winsorized models train... --"
)
if not df_train.empty and len(df_train) > 0:
    scores_df = model_drafts(df_train, scores_df)
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


@st.cache_resource
def tune_and_train_final_model(
    df_train_input, all_params, forecast_period, scores_df_input
):
    rmses = []
    current_scores_df = scores_df_input.copy()
    best_params_dict = {}
    m_final = None
    forecast_final = pd.DataFrame()

    if df_train_input.empty or "y" not in df_train_input.columns:
        st.error(
            "Training data is empty or 'y' column is missing for final model tuning."
        )
        return m_final, current_scores_df, forecast_final, best_params_dict

    for params in all_params:
        try:
            m = Prophet(**params)
            m.fit(df_train_input)
            df_cv = cv_func(m)
            if not df_cv.empty:
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p["rmse"].values[0])
            else:
                st.warning(
                    f"Cross-validation for params {params} returned no results. Skipping metric calculation."
                )
                rmses.append(np.inf)
        except Exception as e:
            st.warning(
                f"Error during tuning with params {params}: {e}. Skipping these parameters."
            )
            rmses.append(np.inf)

    if rmses and min(rmses) != np.inf:
        tuning_results = pd.DataFrame(all_params)
        tuning_results["rmse"] = rmses
        tuning_results = tuning_results.sort_values("rmse").reset_index(drop=True)

        if not tuning_results.empty:
            best_params_dict = dict(tuning_results.drop("rmse", axis="columns").iloc[0])

            m_final = Prophet(**best_params_dict)
            m_final.fit(df_train_input)
            df_cv = cv_func(m_final)
            if not df_cv.empty:
                df_p = performance_metrics(df_cv, rolling_window=1)
                current_scores_df = pd.concat(
                    [current_scores_df, df_p[["mse", "rmse", "mae", "smape"]]],
                    ignore_index=True,
                )
            else:
                st.warning(
                    "Cross-validation for final model returned no results. Skipping metric calculation."
                )
            future = m_final.make_future_dataframe(periods=forecast_period)
            forecast_final = m_final.predict(future)
        else:
            st.error(
                "Tuning results are empty after filtering. Cannot train final model."
            )
    else:
        st.error(
            "No valid tuning results found (all RMSEs were infinite). Cannot train final model."
        )

    return m_final, current_scores_df, forecast_final, best_params_dict


data_load_state = st.text("-- Please wait while the Final Model trains... --")
if not df_train.empty and len(df_train) > 0:
    m, scores_df, forecast, best_params_dict = tune_and_train_final_model(
        df_train, all_params, forecast_period, scores_df
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


# Function & Indicators for Forecasted Line Graph
# Modified parameters to directly use the global forecast_period and train_period if needed for calculation
@st.cache_resource
def plot_forecast(data_to_plot, ticker_name_for_plot, selected_stock_for_plot):
    if data_to_plot.empty:
        st.error("Cannot plot forecast: data_to_plot is empty.")
        return

    fig = go.Figure()

    # Add traces, checking for column existence before adding
    # Always check if the column exists to avoid errors with missing data
    if "yhat_lower" in data_to_plot.columns and "yhat_upper" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["yhat_lower"],
                line=dict(color="lightblue", width=0),
                name="Forecast Lower Bound",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["yhat_upper"],
                line=dict(color="lightblue", width=0),
                name="Forecast Upper Bound",
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.4)",
            )
        )
    else:
        st.warning(
            "Forecast bounds (yhat_lower/yhat_upper) not found in data_to_plot. Skipping these traces."
        )

    if "yhat" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["yhat"],
                line=dict(color="blue", width=2),
                name="Forecast",
                mode="lines",
            )
        )
    else:
        st.warning("Forecast (yhat) not found in data_to_plot. Skipping this trace.")

    if "bb_upper" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["bb_upper"],
                line=dict(color="red", width=1),
                name="Upper BB",
                visible="legendonly",
            )
        )
    else:
        st.warning(
            "Bollinger Band Upper not found in data_to_plot. Skipping this trace."
        )

    if "bb_lower" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["bb_lower"],
                line=dict(color="green", width=1),
                name="Lower BB",
                visible="legendonly",
            )
        )
    else:
        st.warning(
            "Bollinger Band Lower not found in data_to_plot. Skipping this trace."
        )

    if "SMA50" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["SMA50"],
                name="SMA50",
                line=dict(color="black", width=2, dash="dash"),
                visible="legendonly",
            )
        )
    else:
        st.warning("SMA50 not found in data_to_plot. Skipping this trace.")

    if "Adjusted Close" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["Adjusted Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="orange", width=2),
            )
        )
    else:
        st.warning("Adjusted Close not found in data_to_plot. Skipping this trace.")

    fig.layout.update(
        title_text=f"Forecast for Time Series Data: {ticker_name_for_plot} ({selected_stock_for_plot})",
        xaxis_rangeslider_visible=True,
        yaxis_title="Price",
        xaxis_title="Date",
    )

    end_date = data_to_plot["Date"].max()

    # Using a fixed period for initial display or the full length of the data if shorter
    # This avoids reliance on `train_period` or `forecast_period` as function arguments,
    # making `plot_forecast` more self-contained.
    display_period_days = min(
        len(data_to_plot), 365 * 1.5
    )  # Show last 1.5 years or all available data

    start_date = end_date - pd.Timedelta(days=display_period_days)
    start_date = max(
        start_date, data_to_plot["Date"].min()
    )  # Ensure start_date is not before actual data start

    fig.update_xaxes(range=[start_date, end_date])

    st.plotly_chart(fig)


# --- Calling plot_forecast ---
# We are simplifying the plot_forecast function's arguments.
# It now only needs the dataframe and basic display info, not the specific period_unit, train_period, forecast_period.
# The `display_period_days` logic is now internal to `plot_forecast`.
plot_forecast(
    forecast_df, ticker_name, selected_stock
)  # Removed period_unit and train_period from call

st.subheader("-- Chart Tips --")
with st.expander("Click here to expand"):
    st.write("* Use the slider (above) to select a date range")
    st.write("* Click items in the legend to show/hide indicators")
    st.write(
        "* Hover in the upper-right corner of graph to reveal controls. Go fullscreen and explore!"
    )

st.subheader("**-- Accuracy Metrics --**")
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

st.write("-- More Metrics --")
with st.expander("Click here to expand"):
    st.subheader("-- Model Iterations --")
    st.write(
        "The tables below display the performance metrics for each model iteration. The 'Baseline Model' uses the raw closing prices, while the 'Winsorized Model' applies dynamic winsorization to the closing prices. The 'Final Model' is the best-performing model after hyperparameter tuning."
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
            f"* Root Mean Squared Error (RMSE) -  The square root of MSE. It is in the same units as the original data, making it easier to interpret. The RMSE of {round(scores_df.loc['Final Model']['rmse'], 4)} suggests that the model's predictions can deviate from the actual values by up to ${round(scores_df.loc['Final Model']['rmse'], 2)} in some cases."
        )
    else:
        st.write(
            "Detailed metric descriptions are not available due to incomplete model training."
        )

st.subheader("-- About --")
with st.expander("Click here to expand"):
    about_str = f"""
    **-- The Tool --**

    As a passionate trader, I developed this application to streamline my decision-making process. It leverages fundamental data science concepts, including data engineering and analytics, to provide actionable insights. 

    The app features a user-friendly interface with a line chart, Bollinger Bands, and a Simple Moving Average (SMA) for visual analysis of price trends. 

    **-- The Model --**

    To enhance my trading strategy, I've integrated a Prophet forecasting model, fine-tuned with techniques like winsorization and hyperparameter tuning to optimize its accuracy. 

    Key Model Enhancements:
    * Adaptive Winsorization: The winsorization thresholds are dynamically adjusted based on the stock's volatility. 
    * Adaptive Training Data: The training data size is dynamically adjusted based on the stock's volatility and available data.

    By combining these refinements with a cross-validated grid search to optimize changepoint_prior_scale and seasonality_prior_scale, this application provides a robust forecasting tool.

    """
    if best_params_dict:
        about_str += f"For '{selected_stock}', optimal values are: changepoint_prior_scale: {best_params_dict['changepoint_prior_scale']:.3f}, seasonality_prior_scale: {best_params_dict['seasonality_prior_scale']:.3f}.\n\n"
    else:
        about_str += "Optimal hyperparameters could not be determined.\n\n"

    about_str += """
    Cross-validation ensures that the hyperparameters selected are not overfitted to a specific subset of the data. By evaluating the model's performance on multiple subsets of the data during the grid search, we can select hyperparameters that generalize better to unseen data and potentially improve the model's out-of-sample performance. Check out Model Iterations in the More Metrics section (above) to observe the model's improvement over its learning cycles.

    **-- Data Source & Considerations --**

    The core forecasting methodology behind this application was rigorously tested and validated in a prior experiment involving 150 stocks of varying volatility, utilizing comprehensive historical data from `yfinance`. That experiment demonstrated a Symmetric Mean Absolute Percentage Error (SMAPE) of approximately 15% across the diverse dataset, showcasing the model's general effectiveness.

    For this live demonstration, market data is sourced via a free-tier API (Alpha Vantage). Due to API limitations, the available historical data for analysis and forecasting is significantly shortened compared to the original validation experiment. While this allows for a functional demonstration of the forecasting capabilities, **readers should be aware that the limited data history may impact the accuracy and robustness of the forecasts presented within this specific app instance.** The purpose here is to illustrate the application's functionality, not to provide definitive predictive accuracy based on restricted data.

    **-- Trading Tips --**

    By combining the forecasting model with visual aids like a line chart, Bollinger Bands, and SMAs, I'm able to identify potential entry and exit points with greater confidence, ultimately refining my trading decisions.

    By selecting a stock ticker, the app displays important background information like historical highs/lows, percentage change, volatility, and current price alongside the chart. This comprehensive tool empowers more informed trading decisions and refined trading strategies.
    """
    st.write(about_str)

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
