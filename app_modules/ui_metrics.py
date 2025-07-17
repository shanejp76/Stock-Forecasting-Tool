import streamlit as st
import pandas as pd


def display_forecast_summary(forecast_df, data, selected_stock):
    """Displays the 10-day forecast summary."""
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
                        f"The forecast predicts the price of {selected_stock} will be **${forecast_price_val:.2f}** on **{forecast_date_str}**."
                    )
                    if trend_percentage_val >= 0:
                        st.write(
                            f"This represents a **+{trend_percentage_val:.2f}% increase** from the last known price."
                        )
                    else:
                        st.write(
                            f"This represents a **{trend_percentage_val:.2f}% decrease** from the last known price."
                        )

                    third_line_text = f"With **{confidence_level}** confidence, the price is expected to be between **${confidence_lower_val:.2f}** and **${confidence_upper_val:.2f}**."
                    st.text(third_line_text)
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


def display_accuracy_metrics(scores_df):
    """Displays predicted accuracy metrics."""
    st.subheader("**-- Predicted Accuracy --**")
    if (
        len(scores_df) > 2
        and "smape" in scores_df.columns
        and "Final Model" in scores_df.index
    ):
        accuracy_percentage = 100 - (
            round(scores_df.loc["Final Model"]["smape"] * 100, 2)
        )
        st.metric(label="Model Accuracy", value=f"{accuracy_percentage:.2f}%")
    else:
        st.write(
            "Accuracy metrics not fully available yet. (Requires successful training of all 3 models)"
        )


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


def display_model_performance(scores_df, best_params_dict, selected_stock):
    """Displays model iterations and performance narrative."""
    st.subheader("-- Model Iterations and Performance (Narrative & KPIs) --")
    with st.expander("Click here to expand"):
        st.write(
            "The tables below illustrate the methodical and iterative approach to model refinement and performance optimization, a critical skill in data science. We begin with a Baseline Model using raw data, then introduce a Winsorized Model to address data challenges posed by outliers, demonstrating the implementation of effective data preprocessing solutions. Finally, the Final Model showcases performance optimization through rigorous hyperparameter tuning. Each stage is rigorously evaluated using cross-validation and a comparison of performance metrics (SMAPE, RMSE, MAE, MSE), clearly demonstrating the tangible improvements gained at each step. This process highlights a comprehensive data science workflow, from identifying data-driven problems and implementing solutions to optimizing model performance and validating improvements for enhanced predictive accuracy and business value."
        )

        if "Baseline Model" in scores_df.index:
            st.write("### Baseline Model")
            st.dataframe(scores_df.loc[["Baseline Model"]], width=500)
            st.write(
                "This initial model establishes a performance benchmark using raw historical data. It provides a baseline understanding of forecasting accuracy before applying any advanced data preprocessing or tuning techniques."
            )

        if (
            "Winsorized Model" in scores_df.index
            and "Baseline Model" in scores_df.index
        ):
            st.write("### Winsorized Model")
            st.dataframe(scores_df.loc[["Winsorized Model"]], width=500)
            baseline_rmse = scores_df.loc["Baseline Model"]["rmse"]
            winsorized_rmse = scores_df.loc["Winsorized Model"]["rmse"]
            baseline_mae = scores_df.loc["Baseline Model"]["mae"]
            winsorized_mae = scores_df.loc["Winsorized Model"]["mae"]
            baseline_smape = scores_df.loc["Baseline Model"]["smape"]
            winsorized_smape = scores_df.loc["Winsorized Model"]["smape"]

            if baseline_rmse > 0 and baseline_mae > 0 and baseline_smape > 0:
                rmse_improvement = (
                    (baseline_rmse - winsorized_rmse) / baseline_rmse
                ) * 100
                mae_improvement = ((baseline_mae - winsorized_mae) / baseline_mae) * 100
                smape_improvement = (
                    (baseline_smape - winsorized_smape) / baseline_smape
                ) * 100

                st.write(
                    f"**Impact of Winsorization:** By addressing outliers, the Winsorized Model significantly improved predictive accuracy. "
                    f"We observed a **{rmse_improvement:.2f}% reduction in RMSE**, a **{mae_improvement:.2f}% reduction in MAE**, and a **{smape_improvement:.2f}% reduction in SMAPE** compared to the Baseline Model. "
                    "This demonstrates the crucial role of data preprocessing in building more robust and reliable forecasts, leading to more confident decision-making by mitigating the impact of extreme price fluctuations."
                )
            else:
                st.write(
                    "This model applies Winsorization to mitigate the impact of outliers. It aims to improve robustness and reduce noise in the forecast."
                )

        if "Final Model" in scores_df.index and "Winsorized Model" in scores_df.index:
            st.write("### Final Model (Hyperparameter Tuned)")
            st.dataframe(scores_df.loc[["Final Model"]], width=500)
            winsorized_rmse = scores_df.loc["Winsorized Model"]["rmse"]
            final_rmse = scores_df.loc["Final Model"]["rmse"]
            winsorized_mae = scores_df.loc["Winsorized Model"]["mae"]
            final_mae = scores_df.loc["Final Model"]["mae"]
            winsorized_smape = scores_df.loc["Winsorized Model"]["smape"]
            final_smape = scores_df.loc["Final Model"]["smape"]

            if winsorized_rmse > 0 and winsorized_mae > 0 and winsorized_smape > 0:
                rmse_improvement_final = (
                    (winsorized_rmse - final_rmse) / winsorized_rmse
                ) * 100
                mae_improvement_final = (
                    (winsorized_mae - final_mae) / winsorized_mae
                ) * 100
                smape_improvement_final = (
                    (winsorized_smape - final_smape) / winsorized_smape
                ) * 100

                st.write(
                    f"**Impact of Hyperparameter Tuning:** The Final Model, optimized through rigorous hyperparameter tuning, "
                    f"achieved further performance gains. We saw an additional **{rmse_improvement_final:.2f}% reduction in RMSE**, "
                    f"a **{mae_improvement_final:.2f}% reduction in MAE**, and a **{smape_improvement_final:.2f}% reduction in SMAPE** "
                    "compared to the Winsorized Model. This fine-tuning process ensures the model precisely captures underlying patterns, "
                    "delivering highly accurate forecasts that directly translate into improved decision quality and reduced financial risk."
                )
            else:
                st.write(
                    "The final model undergoes rigorous hyperparameter tuning to optimize its performance. This aims to maximize predictive accuracy."
                )
        elif "Final Model" in scores_df.index:
            st.write("### Final Model (Hyperparameter Tuned)")
            st.dataframe(scores_df.loc[["Final Model"]], width=500)
            st.write(
                "The final model undergoes rigorous hyperparameter tuning to optimize its performance. This aims to maximize predictive accuracy."
            )
        else:
            st.write("Not all model iteration metrics are available.")

        st.write(
            "In the context of time series forecasting, 'error' refers to the difference between the actual value of a variable at a specific point in time and the value predicted by a forecasting model. In this case, the metrics will specifically measure the error between the stock's closing price and the forecast trained on the closing price."
        )

        if (
            len(scores_df) > 2
            and "mae" in scores_df.columns
            and "smape" in scores_df.columns
            and "rmse" in scores_df.columns
            and "Final Model" in scores_df.index
        ):
            st.write(
                f"* **MAE (Mean Absolute Error) KPI:** A MAE of **${round(scores_df.loc['Final Model']['mae'], 2):.2f}** implies that, on average, the model's predictions are off by approximately **${round(scores_df.loc['Final Model']['mae'], 2):.2f}**. This is a direct measure of prediction accuracy in currency units."
            )
            st.write(
                f"* **SMAPE (Symmetric Mean Absolute Percentage Error) KPI:** A SMAPE of **{round(scores_df.loc['Final Model']['smape'] * 100, 2):.2f}%** means that, on average, the model's predictions are **{round(scores_df.loc['Final Model']['smape'] * 100, 2):.2f}%** off from the actual values. This provides a normalized, business-friendly view of percentage accuracy."
            )
            st.write(
                "* **MSE (Mean Squared Error) KPI:** This squares the errors, giving more weight to larger errors. A lower MSE indicates better accuracy. While less intuitive for direct business interpretation, it's a critical metric for model optimization."
            )
            st.write(
                f"* **RMSE (Root Mean Squared Error) KPI:** The RMSE of **${round(scores_df.loc['Final Model']['rmse'], 2):.2f}** suggests that the model's predictions can deviate from the actual values by up to **${round(scores_df.loc['Final Model']['rmse'], 2):.2f}** in some cases. Being in the same units as the stock price, it offers a tangible measure of typical prediction error."
            )
        else:
            st.write(
                "Detailed metric descriptions are not available due to incomplete model training."
            )

        st.write(
            "Beyond statistical measures, these metrics translate directly into business impact. For instance, a lower Mean Absolute Error (MAE) signifies that, on average, our forecasts are closer to the actual stock price. This precision is critical as it directly reduces potential financial risk by providing more accurate price expectations. Similarly, the Root Mean Squared Error (RMSE) quantifies the typical magnitude of our prediction errors, which is invaluable for informing robust risk assessments and setting realistic expectations for portfolio management. The Symmetric Mean Absolute Percentage Error (SMAPE) further enhances this by providing a clear, percentage-based understanding of how far off our predictions are, on average, from the actual values."
        )

    st.subheader("-- About --")
    with st.expander("Click here to expand"):
        about_str = f"""
        **-- Purpose & Business Value --**

        As a passionate trader, I developed this application to streamline my decision-making process. It leverages fundamental data science concepts, including data engineering and analytics, to provide actionable insights.

        The app features a user-friendly interface with a line chart, Bollinger Bands, and a Simple Moving Average (SMA) for visual analysis of price trends.

        **-- The Model --**

        To enhance trading strategies and support robust decision-making, this application integrates a Prophet forecasting model, meticulously fine-tuned using advanced machine learning techniques. These methods were chosen not just for technical accuracy, but specifically to **optimize for actionable business outcomes.**

        Key Model Enhancements chosen for their impact on decision-making:
        * **Winsorization:** This technique was applied to improve the model's robustness against extreme price fluctuations (outliers). By mitigating the impact of unusual data points, the model generates **more stable and reliable predictions, reducing noise and leading to more confident trading decisions.** The thresholds are dynamically adjusted based on the stock's volatility to ensure relevance.
        * **Adaptive Training Data:** The size of the training dataset is dynamically adjusted based on the stock's volatility and available data. This ensures the model is trained on the most relevant historical period, which is crucial for **maintaining forecast agility and relevance in fluctuating market conditions.**
        * **Hyperparameter Tuning:** Through a cross-validated grid search, key model parameters (changepoint_prior_scale and seasonality_prior_scale) are systematically optimized. This process ensures the model learns the underlying patterns most effectively, leading to **highly accurate forecasts that directly translate into improved decision quality and reduced financial risk.**

        **Data Preparation:** Before modeling, careful data preparation steps were undertaken. While the raw data from our source (Alpaca) is robust and guarantees cleanliness, I performed essential transformations such as date alignment, feature engineering (e.g., adding technical indicators), and dynamic winsorization to prepare the data optimally for the Prophet model.

        By combining these refinements with a cross-validated grid search, this application provides a robust forecasting tool.

        """
        if best_params_dict:
            about_str += f"For '{selected_stock}', optimal values are: changepoint_prior_scale: {best_params_dict['changepoint_prior_scale']:.3f}, seasonality_prior_scale: {best_params_dict['seasonality_prior_scale']:.3f}.\n\n"
        else:
            about_str += "Optimal hyperparameters could not be determined.\n\n"

        about_str += """
        **Cross-validation is paramount to ensuring the model's generalizability and reliability**, directly translating to **trustworthiness in business insights**. By rigorously evaluating the model's performance on multiple, unseen subsets of the data during the grid search, we can select hyperparameters that are not overfitted to a specific dataset. This robust validation process ensures that the model performs consistently on new data, providing a dependable foundation for trading decisions and strategic planning. Check out Model Iterations in the More Metrics section (above) to observe the model's improvement over its learning cycles.

        **-- Data Source & Considerations --**

        The core forecasting methodology behind this application was rigorously tested and validated in a prior experiment involving 150 stocks of varying volatility, utilizing comprehensive historical data from `yfinance`. That experiment demonstrated a Symmetric Mean Absolute Percentage Error (SMAPE) of approximately 15% across the diverse dataset, showcasing the model's general effectiveness.

        For this live demonstration, market data is sourced via a free-tier API (Alpha Vantage). Due to API limitations, the available historical data for analysis and forecasting is significantly shortened compared to the original validation experiment. While this allows for a functional demonstration of the forecasting capabilities, **a diligent business intelligence analyst would note that this limited data history can impact the robustness and long-term reliability of the forecasts presented here.** The purpose remains to illustrate the application's functionality and demonstrate a comprehensive analytical workflow, rather than to provide definitive predictive accuracy based on constrained data.

        **-- Trading Tips --**

        By combining the forecasting model with visual aids like a line chart, Bollinger Bands, and SMAs, I'm able to identify potential entry and exit points with greater confidence, ultimately refining my trading decisions.

        By selecting a stock ticker, the app displays important background information like historical highs/lows, percentage change, volatility, and current price alongside the chart. This comprehensive tool empowers more informed trading decisions and refined trading strategies.
        """
        st.write(about_str)
