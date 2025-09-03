"""
Business KPIs Module for Swing Ticker

This module provides Streamlit UI components for displaying business intelligence KPIs
related to stock forecasts. It summarizes price movement forecasts, volume trends,
volatility assessment, and market correlation, helping users interpret the forecast
results in a business context.

Functions:
    display_business_kpis(forecast_df, data, stats, volatility, market_correlation):
        Renders the business KPIs section, including price movement, volume, volatility,
        and market correlation metrics.

Author: Shane
Created: 2024-12-04
"""

# app_modules/business_kpis.py
import streamlit as st
import pandas as pd
import numpy as np


def display_business_kpis(
    forecast_df, data, stats, volatility, market_correlation, price_col="Close"
):
    """
    Displays business intelligence KPIs.

    Args:
        forecast_df (pd.DataFrame): DataFrame containing the forecast data.
        data (pd.DataFrame): DataFrame containing the historical stock data.
        stats (dict): Dictionary containing various stock statistics, including 'Annualized Volatility' category.
        volatility (float): Annualized numerical volatility of the stock.
        market_correlation (float or None): Correlation coefficient with a market index.
    """
    st.subheader("-- Business Intelligence KPIs --")
    with st.expander("Click here to expand"):
        if not forecast_df.empty and not data.empty:
            last_actual_price = data[price_col].iloc[-1]
            # Handle Date as either column or index
            if "Date" in data.columns:
                last_actual_date = data["Date"].max()
            else:
                last_actual_date = data.index.max()

            forecast_df_sorted = forecast_df.sort_values(by="Date").reset_index(
                drop=True
            )

            if isinstance(last_actual_date, (pd.Timestamp, pd.DatetimeIndex)):
                comparison_date = last_actual_date
            else:
                comparison_date = pd.to_datetime(last_actual_date)

            forecast_start_idx = forecast_df_sorted[
                forecast_df_sorted["Date"] > comparison_date
            ].index.min()

            if pd.notna(forecast_start_idx):
                st.markdown("### Price Movement Forecasts")
                price_movement_data = {}
                periods = [1, 7, 30]

                for p in periods:
                    target_forecast_idx = int(forecast_start_idx + (p - 1))
                    if target_forecast_idx < len(forecast_df_sorted):
                        forecast_price = forecast_df_sorted.loc[
                            target_forecast_idx, "yhat"
                        ]
                        percentage_change = (
                            (forecast_price - last_actual_price) / last_actual_price
                        ) * 100

                        if percentage_change >= 0:
                            change_text = f"+{percentage_change:.2f}% increase"
                        else:
                            change_text = f"{percentage_change:.2f}% decrease"

                        price_movement_data[f"{p}-Day Forecast Price Change"] = (
                            change_text
                        )
                        price_movement_data[f"{p}-Day Forecast Price"] = (
                            f"${forecast_price:.2f}"
                        )
                    else:
                        price_movement_data[f"{p}-Day Forecast Price Change"] = (
                            "N/A (Forecast too short)"
                        )
                        price_movement_data[f"{p}-Day Forecast Price"] = "N/A"

                price_movement_df = pd.DataFrame.from_dict(
                    price_movement_data, orient="index", columns=["Value"]
                )
                st.dataframe(price_movement_df.T, use_container_width=True)

                st.markdown("### Volume Trends")
                if "Volume" in data.columns and not data.empty:
                    recent_volume = data["Volume"].iloc[-1]
                    average_daily_volume = data["Volume"].mean()

                    volume_data = {}
                    volume_data["Current/Last Known Volume"] = f"{recent_volume:,.0f}"
                    volume_data["Average Daily Volume (Historical)"] = (
                        f"{average_daily_volume:,.0f}"
                    )

                    volume_percentage_difference = np.nan
                    if average_daily_volume != 0:
                        volume_percentage_difference = (
                            (recent_volume - average_daily_volume)
                            / average_daily_volume
                        ) * 100
                        if volume_percentage_difference >= 0:
                            volume_data["% Diff from Average Volume"] = (
                                f"+{volume_percentage_difference:.2f}%"
                            )
                        else:
                            volume_data["% Diff from Average Volume"] = (
                                f"{volume_percentage_difference:.2f}%"
                            )
                    else:
                        volume_data["% Diff from Average Volume"] = (
                            "N/A (Avg Volume is Zero)"
                        )

                    volume_df = pd.DataFrame.from_dict(
                        volume_data, orient="index", columns=["Value"]
                    )
                    st.dataframe(volume_df.T, use_container_width=True)

                    st.markdown(
                        """
                        * **Significance:** Volume indicates market interest and liquidity.
                        * Higher volumes during price movements can confirm the strength of a trend.
                        * A significant difference between recent and average volume might suggest unusual trading activity.
                        * **Interpretation Guidelines:**
                            * **High Positive (e.g., > +20%):** Suggests unusually strong buying or selling interest, often confirming a price trend.
                            * **High Negative (e.g., < -20%):** Indicates declining interest or lower liquidity, which can signal weakening trends.
                            * **Note:** The exact 'significant' threshold can vary greatly depending on the stock's typical liquidity and market conditions.
                        """
                    )
                else:
                    st.write("Volume data not available.")

                st.markdown("### Volatility Assessment")
                if volatility is not None and "Annualized Volatility" in stats:
                    volatility_rank = stats["Annualized Volatility"]

                    st.write(f"**Annualized Volatility:** {volatility:.2%}")
                    st.write(f"**Volatility Rank:** {volatility_rank}")
                    st.markdown(
                        """
                        * **Interpretation Guidelines:**
                            * **Higher volatility** indicates greater price fluctuation risk and potential for larger daily swings.
                        """
                    )
                else:
                    st.write("Volatility data not available.")

                st.markdown("### Market Correlation")
                if market_correlation is not None:
                    st.write(
                        f"**Correlation to S&P 500 (SPY):** {market_correlation:.2f}"
                    )
                    st.markdown(
                        """
                        * A value close to +1 means strong positive correlation, -1 strong negative correlation, 0 no correlation.
                        * **Interpretation Guidelines:**
                            * **Positive (+1):** Stock generally moves *with* the market. Offers limited diversification; exposed to broader market risks.
                            * **Negative (-1):** Stock generally moves *opposite* the market. Can offer diversification/hedging benefits (rare for individual stocks).
                            * **Near Zero (0):** Stock moves *independently* of the market. Good for diversification; performance tied to company-specific factors.
                        """
                    )
                else:
                    st.write(
                        f"**Correlation to S&P 500 (SPY):** Not available (data insufficient or error during calculation)."
                    )

            else:
                st.warning(
                    "Could not calculate BI KPIs: Forecast start index not found."
                )
        else:
            st.warning("BI KPIs not available: forecast_df or data is empty.")
