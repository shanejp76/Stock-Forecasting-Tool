import streamlit as st
import pandas as pd
from plotly import graph_objs as go
from plotly.subplots import make_subplots


@st.cache_resource
def plot_forecast(data_to_plot, ticker_name_for_plot, selected_stock_for_plot):
    """
    Plots the stock forecast along with actual data and technical indicators.
    Uses subplots for MACD and RSI to keep the main price chart clean.
    """
    if data_to_plot.empty:
        st.error("Cannot plot forecast: data_to_plot is empty.")
        return

    # Create subplots: 3 rows, 1 column. Row heights are proportional (e.g., 3 parts price, 1 part RSI, 1 part MACD)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[
            0.6,
            0.2,
            0.2,
        ],  # Adjust heights: 60% for price, 20% for RSI, 20% for MACD
        subplot_titles=(
            f"Price & Forecast for {ticker_name_for_plot} ({selected_stock_for_plot})",
            "Relative Strength Index (RSI)",
            "Moving Average Convergence Divergence (MACD)",
        ),
    )

    # --- TOP SUBPLOT: Price, Forecast, SMAs, Bollinger Bands, Cross Signals ---
    # Add traces in REVERSE order of desired legend appearance (from bottom to top in code)
    # This will make them appear from top to bottom in the actual Plotly legend.
    # IMPORTANT: Traces added here for legend order should have visible="legendonly"
    #            if they are also plotted in a different subplot.

    # 1. MACD Histogram (Now at the very top of the code to appear at the very bottom of the legend)
    if (
        "MACD" in data_to_plot.columns
        and "MACD_Signal" in data_to_plot.columns
        and "MACD_Hist" in data_to_plot.columns
    ):
        colors = ["red" if val < 0 else "green" for val in data_to_plot["MACD_Hist"]]
        fig.add_trace(
            go.Bar(
                x=data_to_plot["Date"],
                y=data_to_plot["MACD_Hist"],
                name="MACD Histogram",
                marker_color=colors,
                visible="legendonly",  # ONLY visible in legend, not on this subplot
            ),
            row=1,
            col=1,
        )
    else:
        st.warning("MACD data not found. Skipping MACD Histogram trace in legend.")

    # 2. MACD Signal
    if (
        "MACD" in data_to_plot.columns
        and "MACD_Signal" in data_to_plot.columns
        and "MACD_Hist" in data_to_plot.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["MACD_Signal"],
                mode="lines",
                name="MACD Signal",
                line=dict(color="orange", width=1),
                visible="legendonly",  # ONLY visible in legend, not on this subplot
            ),
            row=1,
            col=1,
        )
    # No else warning here, as it's covered by the main MACD check

    # 3. MACD
    if (
        "MACD" in data_to_plot.columns
        and "MACD_Signal" in data_to_plot.columns
        and "MACD_Hist" in data_to_plot.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="blue", width=2),
                visible="legendonly",  # ONLY visible in legend, not on this subplot
            ),
            row=1,
            col=1,
        )
    # No else warning here, as it's covered by the main MACD check

    # 4. RSI
    if "RSI" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="purple"),
                visible="legendonly",  # ONLY visible in legend, not on this subplot
            ),
            row=1,
            col=1,
        )
    else:
        st.warning("RSI data not found. Skipping RSI trace in legend.")

    # 5. Death Cross Signal
    if "DeathCross_Signal" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["DeathCross_Signal"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="red"),
                name="Death Cross (SMA50/200)",
                visible="legendonly",
            ),
            row=1,
            col=1,
        )
    else:
        st.warning("Death Cross Signal not found in data_to_plot. Skipping this trace.")

    # 6. Golden Cross Signal
    if "GoldenCross_Signal" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["GoldenCross_Signal"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="green"),
                name="Golden Cross (SMA50/200)",
                visible="legendonly",
            ),
            row=1,
            col=1,
        )
    else:
        st.warning(
            "Golden Cross Signal not found in data_to_plot. Skipping this trace."
        )

    # 7. Lower BB
    if "bb_lower" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["bb_lower"],
                line=dict(color="green", width=1),
                name="Lower BB",
                visible="legendonly",
            ),
            row=1,
            col=1,
        )
    else:
        st.warning(
            "Bollinger Band Lower not found in data_to_plot. Skipping this trace."
        )

    # 8. Upper BB
    if "bb_upper" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["bb_upper"],
                line=dict(color="red", width=1),
                name="Upper BB",
                visible="legendonly",
            ),
            row=1,
            col=1,
        )
    else:
        st.warning(
            "Bollinger Band Upper not found in data_to_plot. Skipping this trace."
        )

    # Reordered SMAs to appear numerically (20, 50, 100, 200) in legend
    # Added in reverse order in code: 200, 100, 50, 20

    # 9. SMA200
    if "SMA200" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["SMA200"],
                name="SMA200",
                line=dict(color="teal", width=1),
                visible="legendonly",
            ),
            row=1,
            col=1,
        )
    else:
        st.warning("SMA200 not found in data_to_plot. Skipping this trace.")

    # 10. SMA100
    if "SMA100" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["SMA100"],
                name="SMA100",
                line=dict(color="purple", width=1),
                visible="legendonly",
            ),
            row=1,
            col=1,
        )
    else:
        st.warning("SMA100 not found in data_to_plot. Skipping this trace.")

    # 11. SMA50 (existing, moved into numerical order)
    if "SMA50" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["SMA50"],
                name="SMA50",
                line=dict(color="black", width=2, dash="dash"),  # Existing style
                visible="legendonly",
            ),
            row=1,
            col=1,
        )
    else:
        st.warning("SMA50 not found in data_to_plot. Skipping this trace.")

    # 12. SMA20 (style changed to dotted black, moved into numerical order)
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
            ),
            row=1,
            col=1,
        )
    else:
        st.warning("SMA20 not found in data_to_plot. Skipping this trace.")

    # 13. Forecast Lower Bound (Part of the fill group)
    if "yhat_lower" in data_to_plot.columns and "yhat_upper" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["yhat_lower"],
                line=dict(color="lightblue", width=0),
                name="Forecast Lower Bound",
                showlegend=False,  # Often hidden as the fill covers it
            ),
            row=1,
            col=1,
        )
    else:
        st.warning(
            "Forecast bounds (yhat_lower/yhat_upper) not found in data_to_plot. Skipping these traces."
        )

    # 14. Forecast Upper Bound (Part of the fill group)
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
            ),
            row=1,
            col=1,
        )
    # No else warning here, as it's covered by the previous check for both lower/upper

    # 15. Add Forecast (yhat)
    if "yhat" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["yhat"],
                line=dict(color="blue", width=2),
                name="Forecast",
                mode="lines",
            ),
            row=1,
            col=1,
        )
    else:
        st.warning("Forecast (yhat) not found in data_to_plot. Skipping this trace.")

    # 16. Add Actual Close Price (moved to the bottom of the code to appear at the top of the legend)
    if "Adjusted Close" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["Adjusted Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="orange", width=2),
            ),
            row=1,
            col=1,
        )
    else:
        st.warning("Adjusted Close not found in data_to_plot. Skipping this trace.")

    # --- MIDDLE SUBPLOT: RSI ---
    # This remains unchanged, showlegend=False is already applied here
    if "RSI" in data_to_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="purple"),
                showlegend=False,  # Hide from the legend in the separate subplot
            ),
            row=2,
            col=1,
        )
        # Add RSI Overbought (70) and Oversold (30) lines
        fig.add_hline(
            y=70,
            line_dash="dot",
            line_color="red",
            annotation_text="Overbought",
            row=2,
            col=1,
        )
        fig.add_hline(
            y=30,
            line_dash="dot",
            line_color="green",
            annotation_text="Oversold",
            row=2,
            col=1,
        )
        fig.update_yaxes(range=[0, 100], row=2, col=1)  # Set fixed range for RSI
    else:
        st.warning("RSI data not found. Skipping RSI plot.")

    # --- BOTTOM SUBPLOT: MACD ---
    # This remains unchanged, showlegend=False is already applied here
    if (
        "MACD" in data_to_plot.columns
        and "MACD_Signal" in data_to_plot.columns
        and "MACD_Hist" in data_to_plot.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="blue", width=2),
                showlegend=False,  # Hide from the legend in the separate subplot
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data_to_plot["Date"],
                y=data_to_plot["MACD_Signal"],
                mode="lines",
                name="MACD Signal",
                line=dict(color="orange", width=1),
                showlegend=False,  # Hide from the legend in the separate subplot
            ),
            row=3,
            col=1,
        )
        # MACD Histogram as bar chart
        colors = ["green"] * len(data_to_plot)
        colors = ["red" if val < 0 else "green" for val in data_to_plot["MACD_Hist"]]
        fig.add_trace(
            go.Bar(
                x=data_to_plot["Date"],
                y=data_to_plot["MACD_Hist"],
                name="MACD Histogram",
                marker_color=colors,
                showlegend=False,  # Hide from the legend in the separate subplot
            ),
            row=3,
            col=1,
        )
    else:
        st.warning("MACD data not found. Skipping MACD plot.")

    # --- Layout Updates for Subplots ---
    fig.update_layout(
        title_text=f"Stock Price Forecast and Indicators for {ticker_name_for_plot} ({selected_stock_for_plot})",
        height=700,  # Increased height to accommodate subplots
        hovermode="x unified",
        legend=dict(
            orientation="v",  # Vertical legend (as per your screenshot)
            yanchor="top",  # Anchor from the top of the legend box
            y=1,  # Position at the top of the figure (relative to paper)
            xanchor="left",  # Anchor from the left of the legend box
            x=1.02,  # Position just outside the right edge of the plot area (relative to paper)
            # Adjust these for more fine-tuning if needed:
            # xref="paper", yref="paper"
        ),
    )

    # Set common x-axis properties and individual y-axis properties
    fig.update_xaxes(
        # The rangeslider should be visible on the last subplot's x-axis to control all
        rangeslider_visible=True,
        row=3,
        col=1,
    )
    # Ensure other x-axes are hidden and synced
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    # Set initial range to display approx 1.5 years by default, synced across all subplots
    end_date = data_to_plot["Date"].max()
    display_period_days = min(len(data_to_plot), 365 * 1.5)
    start_date = end_date - pd.Timedelta(days=display_period_days)
    start_date = max(
        start_date, data_to_plot["Date"].min()
    )  # Ensure start_date isn't before data begins

    # Apply the range to all x-axes
    fig.update_xaxes(range=[start_date, end_date], row=1, col=1)
    fig.update_xaxes(range=[start_date, end_date], row=2, col=1)
    fig.update_xaxes(range=[start_date, end_date], row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)
