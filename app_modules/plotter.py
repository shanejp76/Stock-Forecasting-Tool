# app_modules/plotter.py
import streamlit as st
import pandas as pd
from plotly import graph_objs as go


@st.cache_resource
def plot_forecast(data_to_plot, ticker_name_for_plot, selected_stock_for_plot):
    """
    Plots the stock forecast along with actual data and technical indicators.
    The legend order is determined by the reverse order of adding traces
    to achieve the desired top-to-bottom legend appearance.
    """
    if data_to_plot.empty:
        st.error("Cannot plot forecast: data_to_plot is empty.")
        return

    fig = go.Figure()

    # Add traces in REVERSE order of desired legend appearance (from bottom to top in code)
    # This will make them appear from top to bottom in the actual Plotly legend.

    # 12. Death Cross Signal
    if "DeathCross_Signal" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["DeathCross_Signal"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                name="Death Cross (SMA50/200)",
                visible="legendonly",
            )
        )
    else:
        st.warning("Death Cross Signal not found in data_to_plot. Skipping this trace.")

    # 11. Golden Cross Signal
    if "GoldenCross_Signal" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["GoldenCross_Signal"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="green"),
                name="Golden Cross (SMA50/200)",
                visible="legendonly",
            )
        )
    else:
        st.warning(
            "Golden Cross Signal not found in data_to_plot. Skipping this trace."
        )

    # 10. Lower BB
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

    # 9. Upper BB
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

    # Reordered SMAs to appear numerically (20, 50, 100, 200) in legend
    # Added in reverse order in code: 200, 100, 50, 20

    # 8. SMA200
    if "SMA200" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["SMA200"],
                name="SMA200",
                line=dict(color="teal", width=1),
                visible="legendonly",
            )
        )
    else:
        st.warning("SMA200 not found in data_to_plot. Skipping this trace.")

    # 7. SMA100
    if "SMA100" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["SMA100"],
                name="SMA100",
                line=dict(color="purple", width=1),
                visible="legendonly",
            )
        )
    else:
        st.warning("SMA100 not found in data_to_plot. Skipping this trace.")

    # 6. SMA50 (existing, moved into numerical order)
    if "SMA50" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["SMA50"],
                name="SMA50",
                line=dict(color="black", width=2, dash="dash"),  # Existing style
                visible="legendonly",
            )
        )
    else:
        st.warning("SMA50 not found in data_to_plot. Skipping this trace.")

    # 5. SMA20 (style changed to dotted black, moved into numerical order)
    if "SMA20" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["SMA20"],
                name="SMA20",
                line=dict(
                    color="black", width=1, dash="dot"
                ),  # CHANGED: color and dash
                visible="legendonly",
            )
        )
    else:
        st.warning("SMA20 not found in data_to_plot. Skipping this trace.")

    # 4. Forecast Lower Bound (Part of the fill group)
    if "yhat_lower" in data_to_plot.columns and "yhat_upper" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["yhat_lower"],
                line=dict(color="lightblue", width=0),
                name="Forecast Lower Bound",
                showlegend=False,  # Often hidden as the fill covers it
            )
        )
    else:
        st.warning(
            "Forecast bounds (yhat_lower/yhat_upper) not found in data_to_plot. Skipping these traces."
        )

    # 3. Forecast Upper Bound (Part of the fill group)
    if (
        "yhat_upper" in data_to_plot.columns
    ):  # Only need to check for upper if lower exists
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["yhat_upper"],
                line=dict(color="lightblue", width=0),
                name="Forecast Upper Bound",
                fill="tonexty",  # Fills to the trace before it (yhat_lower)
                fillcolor="rgba(173, 216, 230, 0.4)",
            )
        )
    # No else warning here, as it's covered by the previous check for both lower/upper

    # 2. Add Forecast (yhat)
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

    # 1. Add Actual Close Price (moved to the bottom of the code to appear at the top of the legend)
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
        hovermode="x unified",  # Optional: enhances hover experience
    )

    end_date = data_to_plot["Date"].max()
    display_period_days = min(
        len(data_to_plot), 365 * 1.5
    )  # Display approx 1.5 years by default
    start_date = end_date - pd.Timedelta(days=display_period_days)
    start_date = max(
        start_date, data_to_plot["Date"].min()
    )  # Ensure start_date isn't before data begins

    fig.update_xaxes(range=[start_date, end_date])

    st.plotly_chart(fig, use_container_width=True)
