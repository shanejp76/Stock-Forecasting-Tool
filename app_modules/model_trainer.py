# app_modules/model_trainer.py
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


@st.cache_resource
# Renamed cv_func to _cv_func to prevent hashing
def tune_and_train_final_model(
    df_train_input, all_params, forecast_period, scores_df_input, _cv_func
):
    """
    Tunes hyperparameters, trains the final Prophet model, and generates a forecast.
    """
    rmses = []
    current_scores_df = scores_df_input.copy()
    best_params_dict = {}
    m_final = None
    forecast_final = pd.DataFrame()

    if df_train_input.empty or "y" not in df_train_input.columns:
        st.error(
            "Training data is empty or 'y' column is missing for final model tuning."
        )
        return m_final, current_scores_df, forecast_final, best_params_dict

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
        else:
            st.error(
                "Tuning results are empty after filtering. Cannot train final model."
            )
    else:
        st.error(
            "No valid tuning results found (all RMSEs were infinite). Cannot train final model."
        )

    return m_final, current_scores_df, forecast_final, best_params_dict
