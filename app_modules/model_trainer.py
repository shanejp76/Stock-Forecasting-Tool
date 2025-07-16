import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


def dynamic_winsorize(df, column, window_size=30, percentiles=(0.05, 0.95)):
    """
    Applies dynamic winsorization to a specified column in a DataFrame.
    """
    df["rolling_lower"] = (
        df[column].rolling(window=window_size, min_periods=1).quantile(percentiles[0])
    )
    df["rolling_upper"] = (
        df[column].rolling(window=window_size, min_periods=1).quantile(percentiles[1])
    )
    df["winsorized"] = df[column]
    df.loc[df[column] < df["rolling_lower"], "winsorized"] = df["rolling_lower"]
    df.loc[df[column] > df["rolling_upper"], "winsorized"] = df["rolling_upper"]
    return df


@st.cache_resource
# Renamed cv_func to _cv_func to prevent hashing
def model_drafts(df_train_input, scores_df_input, price_col, _cv_func):
    """
    Trains baseline and winsorized Prophet models and calculates their performance metrics.
    """
    current_scores_df = scores_df_input.copy()
    for col_name in [price_col, "winsorized"]:
        if col_name not in df_train_input.columns:
            st.warning(
                f"Column '{col_name}' not found in training data. Skipping model draft for this column."
            )
            continue

        m = Prophet()
        df_train_renamed = df_train_input[["ds", col_name]].rename(
            columns={col_name: "y"}
        )
        try:
            m.fit(df_train_renamed)
            df_cv = _cv_func(m)  # Use the underscored name here
            if not df_cv.empty:
                df_p = performance_metrics(df_cv, rolling_window=1)
                current_scores_df = pd.concat(
                    [current_scores_df, df_p[["mse", "rmse", "mae", "smape"]]],
                    ignore_index=True,
                )
            else:
                st.warning(
                    f"Cross-validation for {col_name} returned no results. Skipping metric calculation."
                )
        except Exception as e:
            st.error(f"Error training model draft for {col_name}: {e}")
            pass
    return current_scores_df


# --- New Function: prepare_forecast_summary ---
def prepare_forecast_summary(forecast_df, df_train_input, n_days_out):
    """
    Calculates key forecast summary values for a specific number of days out.

    Args:
        forecast_df (pd.DataFrame): The DataFrame containing the Prophet forecast
                                    (including 'ds', 'yhat', 'yhat_lower', 'yhat_upper').
        df_train_input (pd.DataFrame): The training DataFrame, used to get the last actual price
                                       and the last historical date.
        n_days_out (int): The number of days out from the last actual date
                          for which to generate the summary (e.g., 5 for 5 days out).

    Returns:
        dict: A dictionary containing the forecast summary statements.
    """
    summary_data = {
        "forecasted_price_N_days_out": None,
        "forecast_date_N_days_out": None,
        "trend_percentage": None,
        "confidence_lower_N_days_out": None,
        "confidence_upper_N_days_out": None,
    }

    if n_days_out <= 0:
        st.warning(
            "`n_days_out` must be a positive integer for forecast summary. Returning empty summary."
        )
        return summary_data

    if (
        df_train_input.empty
        or "ds" not in df_train_input.columns
        or "y" not in df_train_input.columns
    ):
        st.warning(
            "Training data is empty or missing 'ds'/'y' columns. Cannot prepare forecast summary."
        )
        return summary_data

    if (
        forecast_df.empty
        or "ds" not in forecast_df.columns
        or "yhat" not in forecast_df.columns
    ):
        st.warning(
            "Forecast DataFrame is empty or missing required columns. Cannot prepare forecast summary."
        )
        return summary_data

    # Ensure 'ds' column in both dataframes are datetime
    df_train_input["ds"] = pd.to_datetime(df_train_input["ds"])
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    last_historical_date = df_train_input["ds"].max()
    target_forecast_date = last_historical_date + pd.Timedelta(days=n_days_out)

    # Find the row in `forecast_df` that corresponds to `target_forecast_date`
    # Use `dt.date` for comparison to ignore time components if any exist
    target_row = forecast_df[forecast_df["ds"].dt.date == target_forecast_date.date()]

    if not target_row.empty:
        # If there are multiple entries for the same date (e.g., different times), take the first one
        target_row_data = target_row.iloc[0]
        summary_data["forecasted_price_N_days_out"] = target_row_data["yhat"]
        summary_data["forecast_date_N_days_out"] = target_row_data["ds"].strftime(
            "%Y-%m-%d"
        )
        summary_data["confidence_lower_N_days_out"] = target_row_data["yhat_lower"]
        summary_data["confidence_upper_N_days_out"] = target_row_data["yhat_upper"]

        # Calculate trend: percentage change from last actual price to forecasted price
        last_actual_price = df_train_input["y"].iloc[
            -1
        ]  # Assuming 'y' is the price column and df_train_input is sorted by 'ds'
        if last_actual_price is not None and last_actual_price != 0:
            trend_percentage = (
                (target_row_data["yhat"] - last_actual_price) / last_actual_price
            ) * 100
            summary_data["trend_percentage"] = f"{trend_percentage:.2f}%"
        else:
            summary_data["trend_percentage"] = (
                "N/A (Last actual price is zero or not available)"
            )
    else:
        st.warning(
            f"Could not find forecast for {n_days_out} days out (target date: {target_forecast_date.date()}) in the forecast data."
        )

    return summary_data


# --- End New Function ---


@st.cache_resource
# Renamed cv_func to _cv_func to prevent hashing
def tune_and_train_final_model(
    df_train_input,
    all_params,
    forecast_period,
    scores_df_input,
    _cv_func,
    summary_n_days_out=None,
):
    """
    Tunes hyperparameters, trains the final Prophet model, and generates a forecast.
    Also prepares forecast summary data.

    Args:
        df_train_input (pd.DataFrame): The training DataFrame with 'ds' and 'y' columns.
        all_params (list): List of dictionaries, each representing a set of Prophet model parameters.
        forecast_period (int): The number of days to forecast into the future.
        scores_df_input (pd.DataFrame): DataFrame to store performance metrics.
        _cv_func (function): Cross-validation function (e.g., prophet.diagnostics.cross_validation).
        summary_n_days_out (int, optional): The number of days out for which to generate the forecast summary.
                                            If None, defaults to `forecast_period`.

    Returns:
        tuple: A tuple containing:
            - m_final (Prophet model): The trained final Prophet model.
            - current_scores_df (pd.DataFrame): Updated DataFrame with performance metrics.
            - forecast_final (pd.DataFrame): The final forecast DataFrame (including historical and future).
            - best_params_dict (dict): Dictionary of the best hyperparameters found.
            - forecast_summary (dict): Dictionary of key forecast summary statements.
    """
    rmses = []
    current_scores_df = scores_df_input.copy()
    best_params_dict = {}
    m_final = None
    forecast_final = pd.DataFrame()
    forecast_summary = {}  # Initialize empty summary for robust return

    if df_train_input.empty or "y" not in df_train_input.columns:
        st.error(
            "Training data is empty or 'y' column is missing for final model tuning."
        )
        return (
            m_final,
            current_scores_df,
            forecast_final,
            best_params_dict,
            forecast_summary,
        )

    for params in all_params:
        try:
            m = Prophet(**params)
            m.fit(df_train_input)
            df_cv = _cv_func(m)  # Use the underscored name here
            if not df_cv.empty:
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p["rmse"].values[0])
            else:
                st.warning(
                    f"Cross-validation for params {params} returned no results. Skipping metric calculation."
                )
                rmses.append(np.inf)
        except Exception as e:
            st.warning(
                f"Error during tuning with params {params}: {e}. Skipping these parameters."
            )
            rmses.append(np.inf)

    if rmses and min(rmses) != np.inf:
        tuning_results = pd.DataFrame(all_params)
        tuning_results["rmse"] = rmses
        tuning_results = tuning_results.sort_values("rmse").reset_index(drop=True)

        if not tuning_results.empty:
            best_params_dict = dict(tuning_results.drop("rmse", axis="columns").iloc[0])

            m_final = Prophet(**best_params_dict)
            m_final.fit(df_train_input)
            df_cv = _cv_func(m_final)  # Use the underscored name here
            if not df_cv.empty:
                df_p = performance_metrics(df_cv, rolling_window=1)
                current_scores_df = pd.concat(
                    [current_scores_df, df_p[["mse", "rmse", "mae", "smape"]]],
                    ignore_index=True,
                )
            else:
                st.warning(
                    "Cross-validation for final model returned no results. Skipping metric calculation."
                )
            future = m_final.make_future_dataframe(periods=forecast_period)
            forecast_final = m_final.predict(future)

            # --- Call prepare_forecast_summary ---
            # Determine which 'N' days out to summarize
            actual_summary_n_days_out = (
                summary_n_days_out
                if summary_n_days_out is not None
                else forecast_period
            )

            # Ensure summary_n_days_out doesn't exceed the total forecast period, and is valid
            if actual_summary_n_days_out <= 0:
                st.warning(
                    f"Invalid value for summary_n_days_out: {actual_summary_n_days_out}. Skipping forecast summary generation."
                )
                forecast_summary = {}  # Ensure it's empty
            elif actual_summary_n_days_out > forecast_period:
                st.warning(
                    f"Requested summary for {actual_summary_n_days_out} days out, but forecast period is only {forecast_period} days. Summarizing for {forecast_period} days out instead."
                )
                actual_summary_n_days_out = forecast_period
                forecast_summary = prepare_forecast_summary(
                    forecast_final, df_train_input, actual_summary_n_days_out
                )
            else:
                forecast_summary = prepare_forecast_summary(
                    forecast_final, df_train_input, actual_summary_n_days_out
                )
            # --- End Call prepare_forecast_summary ---

        else:
            st.error(
                "Tuning results are empty after filtering. Cannot train final model."
            )
    else:
        st.error(
            "No valid tuning results found (all RMSEs were infinite). Cannot train final model."
        )

    # --- Updated return values to include forecast_summary ---
    return (
        m_final,
        current_scores_df,
        forecast_final,
        best_params_dict,
        forecast_summary,
    )
