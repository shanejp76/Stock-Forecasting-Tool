"""
Symbol Universe Manager for Stock Forecasting Tool

This module provides functionality to retrieve and manage a comprehensive
universe of stock symbols from Alpha Vantage for bulk data loading.

Functions:
    get_alpha_vantage_symbols(): Get all active symbols from Alpha Vantage

Author: Shane
Created: 2025-08-26
"""

import requests
import pandas as pd
from typing import List, Optional
import logging
import io

logger = logging.getLogger(__name__)


def get_alpha_vantage_symbols(api_key: str, status: str = "active") -> List[str]:
    """
    Retrieve all available symbols from Alpha Vantage LISTING_STATUS endpoint.

    Args:
        api_key: Alpha Vantage API key
        status: 'active' or 'delisted' (default: 'active')

    Returns:
        List[str]: List of stock symbols
    """
    try:
        logger.info(f"Fetching {status} symbols from Alpha Vantage...")

        url = "https://www.alphavantage.co/query"
        params = {"function": "LISTING_STATUS", "apikey": api_key, "state": status}

        response = requests.get(url, params=params)
        response.raise_for_status()

        # Check if response is CSV format
        if "symbol" not in response.text.lower()[:100]:
            logger.error(f"Unexpected response format: {response.text[:200]}")
            return []

        # Parse CSV response
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)

        logger.info(f"Loaded CSV with columns: {df.columns.tolist()}")
        logger.info(f"DataFrame shape: {df.shape}")

        # Get symbols column (case insensitive)
        symbol_col = None
        for col in df.columns:
            if "symbol" in str(col).lower():
                symbol_col = col
                break

        if symbol_col is None:
            logger.error(
                f"No symbol column found. Available columns: {df.columns.tolist()}"
            )
            return []

        # Get symbols and handle NaN values
        symbols = df[symbol_col].dropna().astype(str).tolist()

        # Filter out symbols with special characters that might cause issues
        filtered_symbols = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            # Skip symbols with dots, spaces, or other special characters
            # Also skip very long symbols (likely not regular stocks)
            if (
                len(symbol) <= 5
                and "." not in symbol
                and " " not in symbol
                and symbol.isalpha()
                and symbol != "NAN"
            ):
                filtered_symbols.append(symbol)

        logger.info(
            f"Retrieved {len(filtered_symbols)} {status} symbols from Alpha Vantage"
        )
        logger.info(f"Total symbols before filtering: {len(symbols)}")

        return sorted(list(set(filtered_symbols)))  # Remove duplicates and sort

    except Exception as e:
        logger.error(f"Failed to retrieve symbols from Alpha Vantage: {e}")
        logger.error(
            f"Response content: {response.text[:500] if 'response' in locals() else 'No response'}"
        )
        return []


def get_filtered_symbol_universe(
    api_key: str,
    max_symbols: Optional[int] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Get filtered symbol universe from Alpha Vantage with optional limits.

    Args:
        api_key: Alpha Vantage API key
        max_symbols: Maximum number of symbols to return
        exclude_patterns: Patterns to exclude (e.g., ['WARRANT', 'UNIT'])

    Returns:
        List[str]: Filtered list of symbols
    """
    symbols = get_alpha_vantage_symbols(api_key)

    # Apply exclusion patterns if provided
    if exclude_patterns:
        original_count = len(symbols)
        for pattern in exclude_patterns:
            symbols = [s for s in symbols if pattern.upper() not in s.upper()]
        logger.info(
            f"Excluded {original_count - len(symbols)} symbols with patterns: {exclude_patterns}"
        )

    # Limit number of symbols if specified
    if max_symbols and len(symbols) > max_symbols:
        symbols = symbols[:max_symbols]
        logger.info(f"Limited symbol universe to {max_symbols} symbols")

    logger.info(f"Final filtered symbol universe contains {len(symbols)} symbols")

    return symbols


if __name__ == "__main__":
    # Test the symbol universe functions
    logging.basicConfig(level=logging.INFO)

    # Load API key
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app_modules.config import load_environment_variables

    alpha_vantage_key, _ = load_environment_variables()

    if not alpha_vantage_key:
        print("Error: ALPHA_VANTAGE_API_KEY not found in .env file")
        exit(1)

    print("Testing Alpha Vantage symbol retrieval...")

    # Test symbol retrieval
    symbols = get_alpha_vantage_symbols(alpha_vantage_key)
    print(f"Total symbols: {len(symbols)}")

    if symbols:
        print(f"First 20 symbols: {symbols[:20]}")
        print(f"Last 20 symbols: {symbols[-20:]}")

        # Test filtering
        filtered = get_filtered_symbol_universe(
            alpha_vantage_key,
            max_symbols=100,
            exclude_patterns=["W", "U"],  # Common warrant/unit suffixes
        )
        print(f"Filtered to {len(filtered)} symbols")
    else:
        print("No symbols retrieved")


def get_us_stock_universe() -> List[str]:
    """
    Get the complete universe of US stocks from Alpha Vantage.

    Returns:
        List[str]: List of all active US stock symbols
    """
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not alpha_vantage_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")

    # Get filtered universe (excludes penny stocks, warrants, etc.)
    return get_filtered_symbol_universe(
        alpha_vantage_key,
        max_symbols=None,  # No limit
        exclude_patterns=["W", "U", "."],  # Common suffixes for warrants/units/ADRs
    )
