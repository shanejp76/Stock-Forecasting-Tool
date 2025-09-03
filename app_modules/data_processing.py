"""
Data Processing Module

This module provides functions for processing and cleaning stock data,
including filtering by date ranges and basic data transformations.

Functions:
    process_stock_data(data, start_date): Filter and process raw stock data
    validate_data_integrity(data): Validate data quality and completeness
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
        data: Raw stock data DataFrame with date index
        start_date: Earliest date to include in filtered data

    Returns:
        Filtered and processed DataFrame
    """
    if data.empty:
        return data

    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Filter data by start date
    data = data[data.index >= pd.to_datetime(start_date)]

    # Sort by date ascending
    data = data.sort_index()

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


def validate_data_integrity(data: pd.DataFrame) -> dict:
    """
    Validate data quality and completeness.

    Args:
        data: Stock data DataFrame

    Returns:
        Dictionary with validation results
    """
    if data.empty:
        return {
            "is_valid": False,
            "issues": ["Data is empty"],
            "row_count": 0,
            "date_range": None,
        }

    issues = []

    # Check for required columns
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")

    # Check for negative prices
    price_columns = ["Open", "High", "Low", "Close"]
    for col in price_columns:
        if col in data.columns and (data[col] <= 0).any():
            issues.append(f"Found non-positive values in {col}")

    # Check for logical price relationships
    if all(col in data.columns for col in ["High", "Low", "Open", "Close"]):
        if (data["High"] < data["Low"]).any():
            issues.append("High prices less than Low prices found")
        if ((data["Open"] > data["High"]) | (data["Open"] < data["Low"])).any():
            issues.append("Open prices outside High-Low range found")
        if ((data["Close"] > data["High"]) | (data["Close"] < data["Low"])).any():
            issues.append("Close prices outside High-Low range found")

    # Check for excessive gaps
    if len(data) > 1:
        date_diffs = data.index.to_series().diff().dt.days
        max_gap = date_diffs.max()
        if max_gap > 10:  # More than 10 days gap
            issues.append(f"Found data gap of {max_gap} days")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "row_count": len(data),
        "date_range": (data.index.min(), data.index.max()) if not data.empty else None,
        "missing_values": data.isnull().sum().to_dict(),
    }


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
