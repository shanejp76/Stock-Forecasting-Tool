"""
Data Processing Module

This module provides functions for processing and cleaning stock data,
including filtering by date ranges and basic data transformations.

Functions:
    process_stock_data(data, start_date): Filter and process raw stock data
    clean_stock_data(data): Clean and standardize stock data format

Author: Shane
Created: 2025-09-02 (Refactored from data_handler.py)
"""

import pandas as pd
import numpy as np
from datetime import date
from typing import Optional


def process_stock_data(data: pd.DataFrame, start_date: date) -> pd.DataFrame:
    """
    Filters and processes raw stock data.

    Args:
        data: Raw stock data DataFrame with date column
        start_date: Earliest date to include in filtered data

    Returns:
        Filtered and processed DataFrame
    """
    if data.empty:
        return data

    # Ensure date column exists and is datetime
    if "date" not in data.columns:
        # If date is in index, reset it to column
        if hasattr(data.index, "name") and data.index.name == "date":
            data = data.reset_index()
        else:
            raise ValueError("No 'date' column or index found in data")

    # Ensure date column is datetime
    data["date"] = pd.to_datetime(data["date"])

    # Filter data by start date
    data = data[data["date"] >= pd.to_datetime(start_date)]

    # Sort by date ascending
    data = data.sort_values("date")

    # Remove any rows with missing critical data
    # Check for both capitalized and lowercase column names
    close_col = "Close" if "Close" in data.columns else "close"
    volume_col = "Volume" if "Volume" in data.columns else "volume"

    # Only drop NaN for columns that exist
    columns_to_check = []
    if close_col in data.columns:
        columns_to_check.append(close_col)
    if volume_col in data.columns:
        columns_to_check.append(volume_col)

    if columns_to_check:
        data = data.dropna(subset=columns_to_check)

    return data


def clean_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize stock data format.

    Args:
        data: Raw stock data DataFrame

    Returns:
        Cleaned DataFrame
    """
    if data.empty:
        return data

    # Make a copy to avoid modifying original
    cleaned_data = data.copy()

    # Ensure proper column names (standardize to snake_case)
    column_mapping = {}
    for col in cleaned_data.columns:
        if col.lower() in ["open", "high", "low", "close", "volume"]:
            column_mapping[col] = col.lower()
        elif col.lower() in ["adjusted close", "adj close"]:
            column_mapping[col] = "adjusted_close"

    if column_mapping:
        cleaned_data = cleaned_data.rename(columns=column_mapping)

    # Convert numeric columns to proper types
    numeric_columns = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors="coerce")

    # Remove rows where all price data is missing
    price_cols = [
        col for col in ["open", "high", "low", "close"] if col in cleaned_data.columns
    ]
    if price_cols:
        cleaned_data = cleaned_data.dropna(subset=price_cols, how="all")

    # Forward fill missing values (common for stock data)
    cleaned_data = cleaned_data.fillna(method="ffill")

    # Drop any remaining rows with critical missing data
    if "close" in cleaned_data.columns:
        cleaned_data = cleaned_data.dropna(subset=["close"])

    return cleaned_data
