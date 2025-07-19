"""
Forecast Summary Module for Swing Ticker

This module provides Streamlit UI components for displaying a concise summary
of the 10-day stock price forecast. It calculates the predicted price, percentage
change from the last known price, and confidence interval, helping users quickly
interpret the forecast results.

Functions:
    display_forecast_summary(forecast_df, data, selected_stock):
        Renders the 10-day forecast summary for the selected stock.

Author: Shane
Created: 2024-12-04
"""

# app_modules/forecast_summary.py
import streamlit as st
import pandas as pd


def display_forecast_summary(forecast_df, data, selected_stock):
    """
    Displays the 10-day forecast summary for the selected stock.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing the forecast data.
        data (pd.DataFrame): DataFrame containing the historical stock data.
        selected_stock (str): The ticker symbol of the selected stock.
    """
    if not forecast_df.empty:
        st.markdown("---")
        st.subheader(f"10-Day Forecast Summary for {selected_stock}")

        confidence_level = "80%"
        target_day = 10

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
                        f"Forecast for {target_day} days out is not available in the forecast data. Check forecast_period."
                    )
            else:
                st.warning(
                    "Could not determine the start of the forecast period in the merged data."
                )
        else:
            st.warning(
                f"Forecast data is too short to provide a {target_day}-day summary."
            )
    else:
        st.warning("Forecast summary not available: forecast_df is empty.")
