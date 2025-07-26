"""
Model Orchestrator Module

This module handles the complete model training pipeline, including parameter optimization,
model comparison, and final model training with progress tracking.

Author: Shane
Created: 2025-07-25
"""

import pandas as pd
import streamlit as st
import itertools
from .model_trainer import model_drafts, tune_and_train_final_model


def determine_optimization_strategy(trend_flexibility, seasonality_strength, changepoint_prior, seasonality_prior):
    """
    Determine if using custom parameters or automated optimization.
    
    Args:
        trend_flexibility: User's trend flexibility setting
        seasonality_strength: User's seasonality strength setting
        changepoint_prior: Changepoint prior scale
        seasonality_prior: Seasonality prior scale
        
    Returns:
        tuple: (optimization_mode_description, all_params_list, user_modified_params_flag)
    """
    # Check if user has modified parameters from defaults (slider position 5 = automated)
    user_modified_params = (trend_flexibility != 5) or (seasonality_strength != 5)

    if user_modified_params:
        optimization_mode = "Using your custom parameters. Skipping automated optimization."
        all_params = [
            {
                "changepoint_prior_scale": changepoint_prior,
                "seasonality_prior_scale": seasonality_prior,
            }
        ]
    else:
        optimization_mode = "Running automated parameter optimization."
        param_grid = {
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
        }
        all_params = [
            dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
        ]
    
    return optimization_mode, all_params, user_modified_params


def execute_training_pipeline(df_train, price_col, all_params, forecast_period, run_cross_validation):
    """
    Execute the complete training pipeline with status updates.
    
    Args:
        df_train: Training DataFrame
        price_col: Price column name
        all_params: List of parameter combinations
        forecast_period: Forecast period in days
        run_cross_validation: Cross-validation function
        
    Returns:
        tuple: (model, scores_df, forecast, best_params_dict, forecast_summary, chosen_approach)
    """
    scores_df = pd.DataFrame(columns=["mse", "rmse", "mae", "smape"])
    
    # Initialize return values
    m, forecast, best_params_dict, forecast_summary = None, pd.DataFrame(), {}, ""
    
    # Create a placeholder for progressive status updates
    status_placeholder = st.empty()
    
    # Step 1: Train baseline and winsorized models
    optimization_mode = "Using custom parameters" if len(all_params) == 1 else "Running automated optimization"
    status_placeholder.info(f"Step 1/3: Training baseline models... ({optimization_mode})")
    
    if not df_train.empty and len(df_train) > 0:
        scores_df = model_drafts(
            df_train, scores_df, price_col, _cv_func=run_cross_validation
        )

        # Step 2: Compare models and select best data preparation approach
        status_placeholder.info("Step 2/3: Comparing baseline vs winsorized models...")
        if len(scores_df) >= 2:
            if scores_df.iloc[0]["rmse"] < scores_df.iloc[1]["rmse"]:
                df_train = df_train.rename(columns={price_col: "y"})
                chosen_approach = "raw data"
            else:
                df_train = df_train.rename(columns={"winsorized": "y"})
                chosen_approach = "winsorized data"
        else:
            st.warning(
                "Not enough model drafts for comparison. Using raw 'Adjusted Close' as target for final model."
            )
            df_train = df_train.rename(columns={price_col: "y"})
            chosen_approach = "raw data (fallback)"

        # Step 3: Train final model with hyperparameter tuning
        status_placeholder.info(
            f"Step 3/3: Training final model using {chosen_approach}..."
        )

        m, scores_df, forecast, best_params_dict, forecast_summary = (
            tune_and_train_final_model(
                df_train,
                all_params,
                forecast_period,
                scores_df,
                _cv_func=run_cross_validation,
                summary_n_days_out=forecast_period,
            )
        )

        if m is not None and not forecast.empty:
            status_placeholder.success(
                f"All models trained successfully! Final model uses {chosen_approach}."
            )
        else:
            status_placeholder.error("Error: Model object or forecast is empty")
            st.stop()
    else:
        status_placeholder.error("Error: Training data is empty")
        st.stop()
    
    return m, scores_df, forecast, best_params_dict, forecast_summary, chosen_approach


def format_scores_dataframe(scores_df):
    """
    Format scores DataFrame with proper labels and column ordering.
    
    Args:
        scores_df: DataFrame with model scores
        
    Returns:
        DataFrame: Formatted scores DataFrame
    """
    if len(scores_df) >= 3:
        scores_df.index = ["Baseline Model", "Winsorized Model", "Final Model"]
        scores_df = scores_df.reindex(sorted(scores_df.columns), axis=1)
    
    return scores_df
