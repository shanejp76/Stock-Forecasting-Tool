# app_modules/business_kpis.py
import streamlit as st
import pandas as pd


def display_business_kpis(forecast_df, data, volatility):
    """Displays business intelligence KPIs."""
    st.subheader("-- Business Intelligence KPIs --")
    with st.expander("Click here to expand"):
        if not forecast_df.empty and not data.empty:
            last_actual_price = data["Adjusted Close"].iloc[-1]
            last_actual_date = data["Date"].max()

            forecast_df_sorted = forecast_df.sort_values(by="Date").reset_index(
                drop=True
            )

            forecast_start_idx = forecast_df_sorted[
                forecast_df_sorted["Date"] > last_actual_date
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
                if volatility is not None:
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
                    st.write(f"**Volatility Rank:** {volatility_rank}")
                    st.write(
                        "*(Higher volatility indicates greater price fluctuation risk and potential for larger daily swings.)*"
                    )
                else:
                    st.write("Volatility data not available.")

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
                st.warning(
                    "Could not calculate BI KPIs: Forecast start index not found."
                )
        else:
            st.warning("BI KPIs not available: forecast_df or data is empty.")
