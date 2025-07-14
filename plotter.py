import streamlit as st
import pandas as pd
from plotly import graph_objs as go


@st.cache_resource
def plot_forecast(data_to_plot, ticker_name_for_plot, selected_stock_for_plot):
    """
    Plots the stock forecast along with actual data and technical indicators.
    """
    if data_to_plot.empty:
        st.error("Cannot plot forecast: data_to_plot is empty.")
        return

    fig = go.Figure()

    # Add traces, checking for column existence before adding
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
    display_period_days = min(len(data_to_plot), 365 * 1.5)
    start_date = end_date - pd.Timedelta(days=display_period_days)
    start_date = max(start_date, data_to_plot["Date"].min())

    fig.update_xaxes(range=[start_date, end_date])

    st.plotly_chart(fig)
