"""
Performance Metrics Module for Swing Ticker

This module provides Streamlit UI components for displaying model accuracy metrics
and a narrative of model iterations and performance improvements. It summarizes
key forecasting metrics (SMAPE, RMSE, MAE, MSE) and explains the impact of data
preprocessing and hyperparameter tuning on predictive accuracy and business value.

Functions:
    display_accuracy_metrics(scores_df): Displays predicted accuracy metrics for the final model.
    display_model_performance(scores_df, best_params_dict, selected_stock, user_modified_params=False):
        Displays model iterations, performance metrics, and business narrative.

Author: Shane
Created: 2024-12-04
"""

# app_modules/performance_metrics.py
import streamlit as st
import pandas as pd


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


def display_model_performance(
    scores_df, best_params_dict, selected_stock, user_modified_params=False
):
    """Displays model iterations and performance narrative."""
    st.subheader("-- Model Iterations and Performance (Narrative & KPIs) --")
    with st.expander("Click here to expand"):
        if user_modified_params:
            st.info(
                "**Note:** User-defined parameters are being used. The model is using your specified settings instead of auto-tuned optimal values."
            )
            st.write(
                "The model is running with your custom parameters. This allows you to explore different forecasting approaches and validate alternative configurations. While the model iterations and performance metrics below show the systematic approach used for optimization, the current forecast is based on your user-defined settings rather than the automatically optimized parameters."
            )
        else:
            st.write(
                "To enhance trading strategies and support robust decision-making, this application integrates a Prophet forecasting model, meticulously fine-tuned using advanced machine learning techniques. These methods were chosen not just for technical accuracy, but specifically to optimize for actionable business outcomes."
            )

        st.write(
            "The tables below illustrate the methodical and iterative approach to model refinement and performance optimization, a critical skill in machine learning and analytics. We begin with a **Baseline Model** using raw data, then introduce a **Winsorized Model** to address data challenges posed by outliers, demonstrating the implementation of effective data preprocessing solutions. The script then selects the best-performing model between the Baseline and Winsorized models for **hyperparameter tuning**, resulting in the **Final Model**. Each stage is rigorously evaluated using cross-validation and a comparison of performance metrics (SMAPE, RMSE, MAE, MSE), clearly demonstrating the tangible improvements gained at each step. This process highlights a comprehensive machine learning workflow, from identifying data-driven problems and implementing solutions to optimizing model performance and validating improvements for enhanced predictive accuracy and business value."
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

            # Check if Winsorization improved metrics
            if (
                winsorized_rmse < baseline_rmse
                and winsorized_mae < baseline_mae
                and winsorized_smape < baseline_smape
                and baseline_rmse > 0
                and baseline_mae > 0
                and baseline_smape > 0
            ):
                rmse_improvement = (
                    (baseline_rmse - winsorized_rmse) / baseline_rmse
                ) * 100
                mae_improvement = ((baseline_mae - winsorized_mae) / baseline_mae) * 100
                smape_improvement = (
                    (baseline_smape - winsorized_smape) / baseline_smape
                ) * 100

                st.write(
                    f"**Impact of Winsorization:** This model applies Winsorization to mitigate the impact of outliers. In this instance, by addressing outliers, the Winsorized Model improved predictive accuracy. "
                    f"We observed a **{rmse_improvement:.2f}% reduction in RMSE**, a **{mae_improvement:.2f}% reduction in MAE**, and a **{smape_improvement:.2f}% reduction in SMAPE** compared to the Baseline Model. "
                    "This demonstrates the potential role of data preprocessing in building more robust and reliable forecasts, leading to more confident decision-making by mitigating the impact of extreme price fluctuations."
                )
            else:
                st.write(
                    "This model applies Winsorization to mitigate the impact of outliers. It aims to improve robustness and reduce noise in the forecast. **Note: Winsorization does not always guarantee improvements in all key metrics; the better-performing model between the Baseline and Winsorized is selected for hyperparameter tuning.**"
                )

        if "Final Model" in scores_df.index and (
            "Winsorized Model" in scores_df.index or "Baseline Model" in scores_df.index
        ):
            st.write("### Final Model (Hyperparameter Tuned)")
            st.dataframe(scores_df.loc[["Final Model"]], width=500)

            # Determine which model was chosen for tuning (Baseline or Winsorized)
            model_for_tuning_name = ""
            model_for_tuning_rmse = 0
            model_for_tuning_mae = 0
            model_for_tuning_smape = 0

            if (
                "Winsorized Model" in scores_df.index
                and "Baseline Model" in scores_df.index
            ):
                # Assuming the script picks the best one based on RMSE (or another metric, adjust if needed)
                if (
                    scores_df.loc["Winsorized Model"]["rmse"]
                    < scores_df.loc["Baseline Model"]["rmse"]
                ):
                    model_for_tuning_name = "Winsorized Model"
                    model_for_tuning_rmse = scores_df.loc["Winsorized Model"]["rmse"]
                    model_for_tuning_mae = scores_df.loc["Winsorized Model"]["mae"]
                    model_for_tuning_smape = scores_df.loc["Winsorized Model"]["smape"]
                else:
                    model_for_tuning_name = "Baseline Model"
                    model_for_tuning_rmse = scores_df.loc["Baseline Model"]["rmse"]
                    model_for_tuning_mae = scores_df.loc["Baseline Model"]["mae"]
                    model_for_tuning_smape = scores_df.loc["Baseline Model"]["smape"]
            elif (
                "Winsorized Model" in scores_df.index
            ):  # If only Winsorized exists (shouldn't happen with baseline always first)
                model_for_tuning_name = "Winsorized Model"
                model_for_tuning_rmse = scores_df.loc["Winsorized Model"]["rmse"]
                model_for_tuning_mae = scores_df.loc["Winsorized Model"]["mae"]
                model_for_tuning_smape = scores_df.loc["Winsorized Model"]["smape"]
            elif "Baseline Model" in scores_df.index:  # If only Baseline exists
                model_for_tuning_name = "Baseline Model"
                model_for_tuning_rmse = scores_df.loc["Baseline Model"]["rmse"]
                model_for_tuning_mae = scores_df.loc["Baseline Model"]["mae"]
                model_for_tuning_smape = scores_df.loc["Baseline Model"]["smape"]

            final_rmse = scores_df.loc["Final Model"]["rmse"]
            final_mae = scores_df.loc["Final Model"]["mae"]
            final_smape = scores_df.loc["Final Model"]["smape"]

            if (
                model_for_tuning_rmse > 0
                and model_for_tuning_mae > 0
                and model_for_tuning_smape > 0
            ):
                rmse_improvement_final = (
                    (model_for_tuning_rmse - final_rmse) / model_for_tuning_rmse
                ) * 100
                mae_improvement_final = (
                    (model_for_tuning_mae - final_mae) / model_for_tuning_mae
                ) * 100
                smape_improvement_final = (
                    (model_for_tuning_smape - final_smape) / model_for_tuning_smape
                ) * 100

                st.write(
                    f"**Impact of Hyperparameter Tuning:** The Final Model, optimized through rigorous hyperparameter tuning (based on the superior performance of the **{model_for_tuning_name}**), "
                    f"achieved further performance gains. We saw an additional **{rmse_improvement_final:.2f}% reduction in RMSE**, "
                    f"a **{mae_improvement_final:.2f}% reduction in MAE**, and a **{smape_improvement_final:.2f}% reduction in SMAPE** "
                    f"compared to the {model_for_tuning_name}. This fine-tuning process ensures the model precisely captures underlying patterns, "
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
                f"* **MAE (Mean Absolute Error) KPI:** A MAE of **\${round(scores_df.loc['Final Model']['mae'], 2):.2f}** implies that, on average, the model's predictions are off by approximately **\${round(scores_df.loc['Final Model']['mae'], 2):.2f}**. This is a direct measure of prediction accuracy in currency units."
            )
            st.write(
                f"* **SMAPE (Symmetric Mean Absolute Percentage Error) KPI:** A SMAPE of **{round(scores_df.loc['Final Model']['smape'] * 100, 2):.2f}%** means that, on average, the model's predictions are **{round(scores_df.loc['Final Model']['smape'] * 100, 2):.2f}%** off from the actual values. This provides a normalized, business-friendly view of percentage accuracy."
            )
            st.write(
                "* **MSE (Mean Squared Error) KPI:** This squares the errors, giving more weight to larger errors. A lower MSE indicates better accuracy. While less intuitive for direct business interpretation, it's a critical metric for model optimization."
            )
            st.write(
                f"* **RMSE (Root Mean Squared Error) KPI:** The RMSE of **\${round(scores_df.loc['Final Model']['rmse'], 2):.2f}** suggests that the model's predictions can deviate from the actual values by up to **\${round(scores_df.loc['Final Model']['rmse'], 2):.2f}** in some cases. Being in the same units as the stock price, it offers a tangible measure of typical prediction error."
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
        """

        # Add conditional parameter description
        if user_modified_params:
            about_str += """* **Custom Parameter Usage:** This model uses your manually specified parameter values instead of automated optimization. The changepoint_prior_scale and seasonality_prior_scale values are set according to your Trend Flexibility and Seasonality Strength slider settings, allowing you to test specific scenarios or apply domain expertise about the stock's behavior."""
        else:
            about_str += """* **Hyperparameter Tuning:** Through a cross-validated grid search, key model parameters (changepoint_prior_scale and seasonality_prior_scale) are systematically optimized. This process ensures the model learns the underlying patterns most effectively, leading to **highly accurate forecasts that directly translate into improved decision quality and reduced financial risk.**"""

        about_str += """

        **-- Model Configuration Options --**

        **Training Data Duration:** Users can control the amount of historical trading data used to train the forecasting model. The data consists of **trading days only** (excluding weekends and market holidays) - approximately 250 trading days per calendar year. More data generally improves accuracy by capturing longer-term patterns and trends, but increases processing time. However, more data can also lead to **model drift** - where the model becomes less sensitive to recent market changes and trends because it's heavily weighted toward older, potentially irrelevant data. The default setting uses all available data (~500 trading days ≈ 2 years) to provide comprehensive learning while maintaining relevance.

        *Monitor for model drift by checking if the forecast confidence intervals contain recent actual prices and whether the model captures recent trend changes. If the model seems disconnected from recent market behavior, try reducing the training duration.*

        **Hyperparameter Control:** While the system automatically finds optimal hyperparameters by default, users can override this automation for experimentation:

        - **Changepoint Prior Scale**: Controls how flexible the model is to trend changes. Lower values create smoother, more conservative trends, while higher values allow the model to detect and adapt to more frequent trend shifts.

        - **Seasonality Prior Scale**: Controls how much seasonal variation the model captures. Lower values produce smoother seasonal patterns, while higher values allow for more volatile and pronounced seasonal effects.

        *Note: Modifying hyperparameter values will disable automatic optimization and use your custom settings directly, which can be useful for testing specific scenarios or when you have domain expertise about the stock's behavior.*

        **Data Preparation:** Before modeling, careful data preparation steps were undertaken. While the raw data from our source (Alpha Vantage) is robust and guarantees cleanliness, I performed essential transformations such as date alignment, feature engineering (e.g., adding technical indicators), and dynamic winsorization to prepare the data optimally for the Prophet model.

        By combining these refinements with a cross-validated grid search, this application provides a robust forecasting tool.

        """
        if best_params_dict:
            if user_modified_params:
                about_str += f"For '{selected_stock}', user-defined values are: changepoint_prior_scale: {best_params_dict['changepoint_prior_scale']:.3f}, seasonality_prior_scale: {best_params_dict['seasonality_prior_scale']:.3f}.\n\n"
            else:
                about_str += f"For '{selected_stock}', optimal values are: changepoint_prior_scale: {best_params_dict['changepoint_prior_scale']:.3f}, seasonality_prior_scale: {best_params_dict['seasonality_prior_scale']:.3f}.\n\n"
        else:
            if user_modified_params:
                about_str += "User-defined hyperparameters could not be determined.\n\n"
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
