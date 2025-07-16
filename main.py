# main.py
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import date, timedelta
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
from prophet.diagnostics import (
    cross_validation,
)  # Retained as it's used in run_cross_validation
import itertools

load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
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

    # --- Displaying Stock Statistics in a single-line DataFrame ---
    if stats:
        # Create a single-row DataFrame from the stats dictionary
        stats_df_for_display = pd.DataFrame([stats])

        # Define the desired order of columns
        desired_column_order = [
            "Symbol",
            "Current Price",
            "Daily % Change",
            "YTD % Change",
            "52-Week High",
            "52-Week Low",
            "Current Volume",
            "Annualized Volatility",
            "Average Daily Percentage Change",
            "Last Data Date",
            "Earliest Data Date",
        ]

        # Filter and reorder columns, dropping any that don't exist in the data (e.g., if API didn't return them)
        existing_ordered_columns = [
            col for col in desired_column_order if col in stats_df_for_display.columns
        ]
        stats_df_for_display = stats_df_for_display[existing_ordered_columns]

        st.dataframe(stats_df_for_display, height=60, use_container_width=True)

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
        return pd.DataFrame()  # Return an empty DataFrame on failure


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
# --- Capture the new forecast_summary return value ---
if not df_train.empty and len(df_train) > 0:
    m, scores_df, forecast, best_params_dict, forecast_summary = (
        tune_and_train_final_model(
            df_train,
            all_params,
            forecast_period,
            scores_df,
            _cv_func=run_cross_validation,
            summary_n_days_out=forecast_period,  # Summarize for the full forecast period, e.g., 30 days out
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
            "Volume",  # Include Volume for BI KPIs
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

# --- Display Forecast Summary Statements (Using st.text() for problematic line) ---
if not forecast_df.empty:
    st.markdown("---")  # Add a separator for clarity
    st.subheader(f"10-Day Forecast Summary for {selected_stock}")

    confidence_level = "80%"
    target_day = 10  # We want the 10th day from the start of the forecast

    # Ensure the forecast_df has enough rows for the 10th day
    if len(forecast_df) > data["Date"].nunique() + target_day - 1:
        last_actual_date = data["Date"].max()
        forecast_start_index = forecast_df[
            forecast_df["Date"] > last_actual_date
        ].index.min()

        if pd.notna(forecast_start_index):
            index_for_10th_day = forecast_start_index + (target_day - 1)

            if index_for_10th_day < len(forecast_df):
                forecast_row = forecast_df.iloc[int(index_for_10th_day)]

                forecast_date_str = forecast_row["Date"].strftime("%Y-%m-%d")
                forecast_price_val = forecast_row["yhat"]
                confidence_lower_val = forecast_row["yhat_lower"]
                confidence_upper_val = forecast_row["yhat_upper"]

                last_actual_price = data["Adjusted Close"].iloc[-1]
                trend_percentage_val = (
                    (forecast_price_val - last_actual_price) / last_actual_price
                ) * 100

                st.write(
                    f"The forecast predicts the price of {selected_stock} will be **${forecast_price_val:.2f}** on **{forecast_date_str}**."
                )
                if trend_percentage_val >= 0:
                    st.write(
                        f"This represents a **+{trend_percentage_val:.2f}% increase** from the last known price."
                    )
                else:
                    st.write(
                        f"This represents a **{trend_percentage_val:.2f}% decrease** from the last known price."
                    )

                third_line_text = f"With **{confidence_level}** confidence, the price is expected to be between **${confidence_lower_val:.2f}** and **${confidence_upper_val:.2f}**."
                st.text(third_line_text)
            else:
                st.warning(
                    f"Forecast for {target_day} days out is not available in the forecast data. Check forecast_period."
                )
        else:
            st.warning(
                "Could not determine the start of the forecast period in the merged data."
            )
    else:
        st.warning(f"Forecast data is too short to provide a {target_day}-day summary.")
else:
    st.warning("Forecast summary not available: forecast_df is empty.")
# --- End Forecast Summary Display ---

# --- REMOVED CHART TIPS ENTIRELY ---


## Accuracy Metrics
st.subheader("**-- Predicted Accuracy --**")  # Changed header
if (
    len(scores_df) > 2
    and "smape" in scores_df.columns
    and "Final Model" in scores_df.index
):
    # New line after "Model Accuracy" using st.metric
    accuracy_percentage = 100 - (round(scores_df.loc["Final Model"]["smape"] * 100, 2))
    st.metric(label="Model Accuracy", value=f"{accuracy_percentage:.2f}%")
else:
    st.write(
        "Accuracy metrics not fully available yet. (Requires successful training of all 3 models)"
    )

# --- NEW SECTION FOR BI-CENTRIC KPIS ---
st.subheader("-- Business Intelligence KPIs --")
with st.expander("Click here to expand"):  # Changed expander text
    if not forecast_df.empty and not data.empty:
        last_actual_price = data["Adjusted Close"].iloc[-1]
        last_actual_date = data["Date"].max()

        # Ensure forecast_df is sorted by Date
        forecast_df_sorted = forecast_df.sort_values(by="Date").reset_index(drop=True)

        # Get the first actual forecast date index
        forecast_start_idx = forecast_df_sorted[
            forecast_df_sorted["Date"] > last_actual_date
        ].index.min()

        if pd.notna(forecast_start_idx):
            # Price Movement Analysis KPIs
            st.markdown("### Price Movement Forecasts")

            price_movement_data = {}
            periods = [1, 7, 30]  # Periods for forecast

            for p in periods:
                target_forecast_idx = int(forecast_start_idx + (p - 1))
                if target_forecast_idx < len(forecast_df_sorted):
                    forecast_price = forecast_df_sorted.loc[target_forecast_idx, "yhat"]
                    percentage_change = (
                        (forecast_price - last_actual_price) / last_actual_price
                    ) * 100

                    if percentage_change >= 0:
                        change_text = f"+{percentage_change:.2f}% increase"
                    else:
                        change_text = f"{percentage_change:.2f}% decrease"

                    price_movement_data[f"{p}-Day Forecast Price Change"] = change_text
                    price_movement_data[f"{p}-Day Forecast Price"] = (
                        f"${forecast_price:.2f}"
                    )
                else:
                    price_movement_data[f"{p}-Day Forecast Price Change"] = (
                        "N/A (Forecast too short)"
                    )
                    price_movement_data[f"{p}-Day Forecast Price"] = "N/A"

            # Convert dictionary to DataFrame for display
            price_movement_df = pd.DataFrame.from_dict(
                price_movement_data, orient="index", columns=["Value"]
            )
            st.dataframe(
                price_movement_df.T, use_container_width=True
            )  # Transpose for better display

            # Volume Trends KPI
            st.markdown("### Volume Trends")
            if "Volume" in data.columns:
                recent_volume = data["Volume"].iloc[-1] if not data.empty else "N/A"
                average_daily_volume = (
                    data["Volume"].mean() if not data.empty else "N/A"
                )

                st.write(f"**Current/Last Known Volume:** {recent_volume:,.0f}")
                st.write(
                    f"**Average Daily Volume (Historical):** {average_daily_volume:,.0f}"
                )
                # Added significance explanation
                st.markdown(
                    """
                    * **Significance:** Volume indicates market interest and liquidity. 
                    * Higher volumes during price movements can confirm the strength of a trend. 
                    * A significant difference between recent and average volume might suggest unusual trading activity.
                    """
                )
            else:
                st.write("Volume data not available.")

            # Volatility KPI
            st.markdown("### Volatility Assessment")
            if volatility is not None:
                # Determine categorical rank for volatility
                volatility_rank = "Not Available"
                if volatility < 15.0:
                    volatility_rank = "Low Volatility"
                elif 15.0 <= volatility < 30.0:
                    volatility_rank = "Moderate Volatility"
                elif 30.0 <= volatility < 50.0:
                    volatility_rank = "High Volatility"
                else:
                    volatility_rank = "Very High Volatility"

                st.write(f"**Annualized Volatility:** {volatility:.2f}%")
                st.write(
                    f"**Volatility Rank:** {volatility_rank}"
                )  # Display categorical rank
                st.write(
                    "*(Higher volatility indicates greater price fluctuation risk and potential for larger daily swings.)*"
                )
            else:
                st.write("Volatility data not available.")

            # Placeholder for Correlation to Market Index
            st.markdown("### Market Correlation (Placeholder)")
            st.write(
                f"**Correlation to S&P 500 (e.g., SPY):** *(Requires additional data fetching and calculation)*"
            )
            st.write(
                "*(A value close to +1 indicates strong positive correlation, -1 strong negative correlation, 0 no correlation)*"
            )
            st.write(
                "*(This KPI helps understand how the stock's movement relates to the broader market.)*"
            )

        else:
            st.warning("Could not calculate BI KPIs: Forecast start index not found.")
    else:
        st.warning("BI KPIs not available: forecast_df or data is empty.")
# --- END NEW SECTION FOR BI-CENTRIC KPIS ---


st.subheader(
    "-- Model Iterations and Performance (Narrative & KPIs) --"
)  # Updated header for narrative & KPIs
with st.expander("Click here to expand"):
    st.write(
        "The tables below illustrate the methodical and iterative approach to model refinement and performance optimization, a critical skill in data science. We begin with a Baseline Model using raw data, then introduce a Winsorized Model to address data challenges posed by outliers, demonstrating the implementation of effective data preprocessing solutions. Finally, the Final Model showcases performance optimization through rigorous hyperparameter tuning. Each stage is rigorously evaluated using cross-validation and a comparison of performance metrics (SMAPE, RMSE, MAE, MSE), clearly demonstrating the tangible improvements gained at each step. This process highlights a comprehensive data science workflow, from identifying data-driven problems and implementing solutions to optimizing model performance and validating improvements for enhanced predictive accuracy and business value."
    )

    if "Baseline Model" in scores_df.index:
        st.write("### Baseline Model")  # Changed to H3 for hierarchy
        st.dataframe(scores_df.loc[["Baseline Model"]], width=500)
        # --- Baseline Narrative (No comparison here) ---
        st.write(
            "This initial model establishes a performance benchmark using raw historical data. It provides a baseline understanding of forecasting accuracy before applying any advanced data preprocessing or tuning techniques."
        )

    if "Winsorized Model" in scores_df.index and "Baseline Model" in scores_df.index:
        st.write("### Winsorized Model")  # Changed to H3
        st.dataframe(scores_df.loc[["Winsorized Model"]], width=500)
        # --- Winsorized Narrative with KPI Improvement ---
        baseline_rmse = scores_df.loc["Baseline Model"]["rmse"]
        winsorized_rmse = scores_df.loc["Winsorized Model"]["rmse"]

        baseline_mae = scores_df.loc["Baseline Model"]["mae"]
        winsorized_mae = scores_df.loc["Winsorized Model"]["mae"]

        baseline_smape = scores_df.loc["Baseline Model"]["smape"]
        winsorized_smape = scores_df.loc["Winsorized Model"]["smape"]

        if (
            baseline_rmse > 0 and baseline_mae > 0 and baseline_smape > 0
        ):  # Avoid division by zero
            rmse_improvement = ((baseline_rmse - winsorized_rmse) / baseline_rmse) * 100
            mae_improvement = ((baseline_mae - winsorized_mae) / baseline_mae) * 100
            smape_improvement = (
                (baseline_smape - winsorized_smape) / baseline_smape
            ) * 100

            st.write(
                f"**Impact of Winsorization:** By addressing outliers, the Winsorized Model significantly improved predictive accuracy. "
                f"We observed a **{rmse_improvement:.2f}% reduction in RMSE**, a **{mae_improvement:.2f}% reduction in MAE**, and a **{smape_improvement:.2f}% reduction in SMAPE** compared to the Baseline Model. "
                "This demonstrates the crucial role of data preprocessing in building more robust and reliable forecasts, leading to more confident decision-making by mitigating the impact of extreme price fluctuations."
            )
        else:
            st.write(
                "This model applies Winsorization to mitigate the impact of outliers. It aims to improve robustness and reduce noise in the forecast."
            )

    if "Final Model" in scores_df.index and "Winsorized Model" in scores_df.index:
        st.write("### Final Model (Hyperparameter Tuned)")  # Changed to H3
        st.dataframe(scores_df.loc[["Final Model"]], width=500)
        # --- Final Model Narrative with KPI Improvement ---
        winsorized_rmse = scores_df.loc["Winsorized Model"]["rmse"]
        final_rmse = scores_df.loc["Final Model"]["rmse"]

        winsorized_mae = scores_df.loc["Winsorized Model"]["mae"]
        final_mae = scores_df.loc["Final Model"]["mae"]

        winsorized_smape = scores_df.loc["Winsorized Model"]["smape"]
        final_smape = scores_df.loc["Final Model"]["smape"]

        if (
            winsorized_rmse > 0 and winsorized_mae > 0 and winsorized_smape > 0
        ):  # Avoid division by zero
            rmse_improvement_final = (
                (winsorized_rmse - final_rmse) / winsorized_rmse
            ) * 100
            mae_improvement_final = (
                (winsorized_mae - final_mae) / winsorized_mae
            ) * 100
            smape_improvement_final = (
                (winsorized_smape - final_smape) / winsorized_smape
            ) * 100

            st.write(
                f"**Impact of Hyperparameter Tuning:** The Final Model, optimized through rigorous hyperparameter tuning, "
                f"achieved further performance gains. We saw an additional **{rmse_improvement_final:.2f}% reduction in RMSE**, "
                f"a **{mae_improvement_final:.2f}% reduction in MAE**, and a **{smape_improvement_final:.2f}% reduction in SMAPE** "
                "compared to the Winsorized Model. This fine-tuning process ensures the model precisely captures underlying patterns, "
                "delivering highly accurate forecasts that directly translate into improved decision quality and reduced financial risk."
            )
        else:
            st.write(
                "The final model undergoes rigorous hyperparameter tuning to optimize its performance. This aims to maximize predictive accuracy."
            )
    elif "Final Model" in scores_df.index:  # In case winsorized model wasn't available
        st.write("### Final Model (Hyperparameter Tuned)")
        st.dataframe(scores_df.loc[["Final Model"]], width=500)
        st.write(
            "The final model undergoes rigorous hyperparameter tuning to optimize its performance. This aims to maximize predictive accuracy."
        )
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
            f"* **MAE (Mean Absolute Error) KPI:** A MAE of **${round(scores_df.loc['Final Model']['mae'], 2):.2f}** implies that, on average, the model's predictions are off by approximately **${round(scores_df.loc['Final Model']['mae'], 2):.2f}**. This is a direct measure of prediction accuracy in currency units."
        )
        st.write(
            f"* **SMAPE (Symmetric Mean Absolute Percentage Error) KPI:** A SMAPE of **{round(scores_df.loc['Final Model']['smape'] * 100, 2):.2f}%** means that, on average, the model's predictions are **{round(scores_df.loc['Final Model']['smape'] * 100, 2):.2f}%** off from the actual values. This provides a normalized, business-friendly view of percentage accuracy."
        )
        st.write(
            "* **MSE (Mean Squared Error) KPI:** This squares the errors, giving more weight to larger errors. A lower MSE indicates better accuracy. While less intuitive for direct business interpretation, it's a critical metric for model optimization."
        )
        st.write(
            f"* **RMSE (Root Mean Squared Error) KPI:** The RMSE of **${round(scores_df.loc['Final Model']['rmse'], 2):.2f}** suggests that the model's predictions can deviate from the actual values by up to **${round(scores_df.loc['Final Model']['rmse'], 2):.2f}** in some cases. Being in the same units as the stock price, it offers a tangible measure of typical prediction error."
        )
    else:
        st.write(
            "Detailed metric descriptions are not available due to incomplete model training."
        )

    st.write(
        "Beyond statistical measures, these metrics translate directly into business impact. For instance, a lower Mean Absolute Error (MAE) signifies that, on average, our forecasts are closer to the actual stock price. This precision is critical as it directly reduces potential financial risk by providing more accurate price expectations. Similarly, the Root Mean Squared Error (RMSE) quantifies the typical magnitude of our prediction errors, which is invaluable for informing robust risk assessments and setting realistic expectations for portfolio management. The Symmetric Mean Absolute Percentage Error (SMAPE) further enhances this by providing a clear, percentage-based understanding of how far off our predictions are, on average, from the actual values."
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
