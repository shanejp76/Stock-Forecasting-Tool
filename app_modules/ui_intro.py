import streamlit as st
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from datetime import date, timedelta

# Import functions from our other modules for data handling
from app_modules.data_handler import (
    load_finnhub_tickers,
    load_alpha_vantage_data,
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


def display_stock_selection(FINNHUB_API_KEY, EXCHANGE_CODE, ts_av):
    """
    Handles stock selection UI and data loading/initial processing.
    Returns: selected_stock, ticker_name, data, stats, percentiles, volatility,
             period_unit, forecast_period, train_period, tickers_data
    """
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

        data = pd.DataFrame()  # Initialize data to an empty DataFrame
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
            stats_df_for_display = pd.DataFrame([stats])
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
            existing_ordered_columns = [
                col
                for col in desired_column_order
                if col in stats_df_for_display.columns
            ]
            stats_df_for_display = stats_df_for_display[existing_ordered_columns]
            st.dataframe(stats_df_for_display, height=60, use_container_width=True)
        else:
            st.warning("Stock statistics are not available.")

        # Determine training and forecast periods
        period_unit, forecast_period, train_period = determine_periods(data, volatility)

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
    )
