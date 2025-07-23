"""
UI Appendix Module for Swing Ticker

This module provides Streamlit UI components for displaying the appendix section
of the Swing Ticker application. It includes expandable sections for the ticker list,
forecast components, forecast grid, and filtered raw data, offering users additional
context and transparency about the forecasting process and underlying data.

Functions:
    display_appendix(tickers_data, m, forecast, data): Renders the appendix section
        with ticker list, forecast components, forecast grid, and filtered raw data.

Author: Shane
Created: 2024-12-04
"""

import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Configure matplotlib to use a safe backend and font
matplotlib.use('Agg')  # Use non-interactive backend

# Try multiple font fallbacks to avoid font loading issues
try:
    plt.rcParams.update({
        'font.family': ['DejaVu Sans', 'Arial', 'sans-serif'],
        'font.size': 10,
        'axes.unicode_minus': False,
        'figure.max_open_warning': 0,
    })
except Exception:
    # Minimal fallback configuration
    plt.rcParams.update({
        'font.size': 10,
        'axes.unicode_minus': False,
    })


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
                
                # Configure matplotlib to avoid font issues completely
                import os
                import tempfile
                
                # Set environment variables to disable font caching
                os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
                
                # Use the most basic matplotlib configuration
                matplotlib.rcdefaults()  # Reset to defaults first
                matplotlib.use('Agg')    # Non-interactive backend
                
                # Set minimal font configuration
                plt.rcParams.update({
                    'font.family': 'sans-serif',
                    'font.sans-serif': ['Arial', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
                    'font.size': 8,
                    'axes.titlesize': 10,
                    'axes.labelsize': 8,
                    'xtick.labelsize': 7,
                    'ytick.labelsize': 7,
                    'legend.fontsize': 8,
                    'figure.titlesize': 12,
                    'axes.unicode_minus': False,
                    'figure.max_open_warning': 0,
                    'figure.facecolor': 'white',
                    'axes.facecolor': 'white'
                })
                
                # Create the plot with Prophet's plot_components
                fig2 = m.plot_components(forecast, figsize=(12, 8))
                
                # Ensure proper figure formatting
                if hasattr(fig2, 'patch'):
                    fig2.patch.set_facecolor('white')
                
                # Display the plot in Streamlit
                st.pyplot(fig2, clear_figure=True)
                
                # Clean up matplotlib resources
                plt.close(fig2)
                plt.clf()
                
            except Exception as e:
                st.error(f"Error plotting forecast components: {str(e)}")
                st.write("**Alternative: Forecast Components Data**")
                
                # Fallback to data display if plotting still fails
                try:
                    components_df = forecast[['ds', 'yhat', 'trend']].copy()
                    components_df['ds'] = pd.to_datetime(components_df['ds'])
                    
                    # Add seasonal components if they exist
                    if 'weekly' in forecast.columns:
                        components_df['weekly_seasonality'] = forecast['weekly']
                    if 'yearly' in forecast.columns:
                        components_df['yearly_seasonality'] = forecast['yearly']
                    if 'holidays' in forecast.columns:
                        components_df['holiday_effects'] = forecast['holidays']
                        
                    # Display recent components data
                    st.dataframe(components_df.tail(14), use_container_width=True)
                    st.write("� The data above shows the same component breakdown that would appear in the visual charts.")
                    
                except Exception as fallback_error:
                    st.write(f"Components data also unavailable: {fallback_error}")
                    
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
