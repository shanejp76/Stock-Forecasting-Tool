"""
Forecast Summary Module for Swing Ticker

This module provides Streamlit UI components for displaying a concise summary
of the stock price forecast. It adaptively shows up to a 10-day forecast summary
when available, or the maximum available forecast period for shorter forecasts.
It calculates the predicted price, percentage change from the last known price,
and confidence interval, helping users quickly interpret the forecast results.

Functions:
    display_forecast_summary(forecast_df, data, selected_stock):
        Renders an adaptive forecast summary for the selected stock.

Author: Shane
Created: 2024-12-04
Updated: 2025-07-21 - Made forecast summary adaptive to training duration
"""

# app_modules/forecast_summary.py
import streamlit as st
import pandas as pd


def display_forecast_summary(
    forecast_df, data, selected_stock, scores_df=None, price_col="Close"
):
    """
    Displays an adaptive forecast summary for the selected stock.
    Shows forecast for 10 days if available, otherwise shows the maximum available forecast period.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing the forecast data.
        data (pd.DataFrame): DataFrame containing the historical stock data.
        selected_stock (str): The ticker symbol of the selected stock.
        scores_df (pd.DataFrame, optional): DataFrame containing model accuracy scores.
    """
    if not forecast_df.empty:
        st.markdown("---")

        confidence_level = "80%"
        preferred_target_day = 10  # Preferred forecast day

        # Find the actual forecast data (after historical data ends)
        # Handle date being in index or column
        if "Date" in data.columns:
            last_actual_date = data["Date"].max()
        else:
            # Date is in the index
            last_actual_date = data.index.max()

        forecast_only_df = forecast_df[forecast_df["Date"] > last_actual_date]

        if len(forecast_only_df) > 0:
            # Determine the actual target day based on available forecast data
            max_available_forecast_days = len(forecast_only_df)
            target_day = min(preferred_target_day, max_available_forecast_days)

            # Update the header to show the actual forecast period being displayed
            if target_day == preferred_target_day:
                st.subheader(
                    f"{target_day}-Trading-Day Forecast Summary for {selected_stock}"
                )
            else:
                st.subheader(
                    f"{target_day}-Trading-Day Forecast Summary for {selected_stock}"
                )
                st.info(
                    f"Note: Showing {target_day} trading days forecast (maximum available with current settings). Increase training duration for longer forecasts."
                )

            # Get the forecast row for the target day
            forecast_row = forecast_only_df.iloc[
                target_day - 1
            ]  # -1 because index starts at 0

            forecast_date_str = forecast_row["Date"].strftime("%Y-%m-%d")
            forecast_price_val = forecast_row["yhat"]
            confidence_lower_val = forecast_row["yhat_lower"]
            confidence_upper_val = forecast_row["yhat_upper"]

            last_actual_price = data[price_col].iloc[-1]
            trend_percentage_val = (
                (forecast_price_val - last_actual_price) / last_actual_price
            ) * 100

            st.write(
                f"The forecast predicts the price of {selected_stock} will be **\${forecast_price_val:.2f}** on **{forecast_date_str}**."
            )

            if trend_percentage_val >= 0:
                st.write(
                    f"This represents a **+{trend_percentage_val:.2f}% increase** from the last known price."
                )
            else:
                st.write(
                    f"This represents a **{trend_percentage_val:.2f}% decrease** from the last known price."
                )

            third_line_text = f"With **{confidence_level}** confidence, the price is expected to be between **\${confidence_lower_val:.2f}** and **\${confidence_upper_val:.2f}**."
            st.markdown(third_line_text)

        else:
            st.warning(
                "No forecast data available beyond the historical data. Please check your model configuration."
            )
    else:
        st.warning("Forecast summary not available: forecast_df is empty.")
