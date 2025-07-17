# app_modules/ui_appendix.py
import streamlit as st
import pandas as pd


def display_appendix(tickers_data, m, forecast, data):
    """Displays the appendix sections."""
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
