# app_modules/business_kpis.py
import streamlit as st
import pandas as pd


# MODIFIED: Added 'stats' to the function arguments
def display_business_kpis(forecast_df, data, stats, volatility, market_correlation):
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
            last_actual_price = data["Adjusted Close"].iloc[-1]
            last_actual_date = data["Date"].max()

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
                if "Volume" in data.columns:
                    recent_volume = data["Volume"].iloc[-1] if not data.empty else "N/A"
                    average_daily_volume = (
                        data["Volume"].mean() if not data.empty else "N/A"
                    )

                    st.write(f"**Current/Last Known Volume:** {recent_volume:,.0f}")
                    st.write(
                        f"**Average Daily Volume (Historical):** {average_daily_volume:,.0f}"
                    )
                    st.markdown(
                        """
                        * **Significance:** Volume indicates market interest and liquidity.
                        * Higher volumes during price movements can confirm the strength of a trend.
                        * A significant difference between recent and average volume might suggest unusual trading activity.
                        """
                    )
                else:
                    st.write("Volume data not available.")

                st.markdown("### Volatility Assessment")
                if (
                    volatility is not None and "Annualized Volatility" in stats
                ):  # Check if stats contains the key
                    # Use the pre-calculated rank from stats
                    volatility_rank = stats["Annualized Volatility"]

                    st.write(f"**Annualized Volatility:** {volatility:.2%}")
                    st.write(f"**Volatility Rank:** {volatility_rank}")
                    st.write(
                        "*(Higher volatility indicates greater price fluctuation risk and potential for larger daily swings.)*"
                    )
                else:
                    st.write("Volatility data not available.")

                # --- NEW CONTENT: Market Correlation ---
                st.markdown("### Market Correlation")
                if market_correlation is not None:
                    st.write(
                        f"**Correlation to S&P 500 (SPY):** {market_correlation:.2f}"
                    )
                    st.write(
                        "*(A value close to +1 indicates strong positive correlation, -1 strong negative correlation, 0 no correlation)*"
                    )
                    st.write(
                        "*(This KPI helps understand how the stock's movement relates to the broader market.)*"
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
