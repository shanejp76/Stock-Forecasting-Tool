"""
Configuration Module for Swing Ticker

This module handles loading environment variables (API keys) and provides
application-wide constants such as the exchange code.

Functions:
    load_environment_variables(): Loads API keys from .env file.

Constants:
    EXCHANGE_CODE (str): Default exchange code for stock data.

Author: Shane
Created: 2024-12-04
"""

# app_modules/config.py
import os
from dotenv import load_dotenv


def load_environment_variables():
    """
    Loads environment variables from a .env file and returns API keys.

    Returns:
        tuple: (alpha_vantage_key, finnhub_api_key)
    """
    load_dotenv()
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
    finnhub_api_key = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
    return alpha_vantage_key, finnhub_api_key


# Application Constants
EXCHANGE_CODE = "US"
