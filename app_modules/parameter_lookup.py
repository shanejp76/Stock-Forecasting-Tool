"""
Parameter Lookup Module for Prophet Optimization

This module provides functions to look up optimal Prophet parameters for symbols,
with fallbacks to generic optimization when symbol-specific parameters are unavailable.

It integrates with the existing UI components to provide seamless parameter loading
while maintaining user override capabilities via sliders.

Key Functions:
- get_optimal_parameters_for_symbol(): Main lookup function with fallbacks
- get_default_parameters(): Generic fallback parameters
- should_use_optimal_parameters(): Determines when to use stored parameters
- update_ui_with_optimal_parameters(): Updates Streamlit UI components

Author: Shane
Created: 2025-09-21
"""

import logging
from typing import Dict, Optional, Tuple
import streamlit as st
import pandas as pd
from app_modules.optimal_parameters import get_optimal_parameters_manager

logger = logging.getLogger(__name__)

# Default Prophet parameters (fallback when no optimization is available)
DEFAULT_PROPHET_PARAMS = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "rmse": None,
    "mae": None,
    "smape": None,
    "source": "default",
}

# Prophet parameter ranges for UI sliders (from existing UI)
PARAM_RANGES = {
    "changepoint_prior_scale": {
        "min": 0.001,
        "max": 1.0,
        "step": 0.001,
        "format": "%.3f",
    },
    "seasonality_prior_scale": {
        "min": 0.01,
        "max": 100.0,
        "step": 0.01,
        "format": "%.2f",
    },
}


class ParameterLookupService:
    """Service for looking up and managing optimal Prophet parameters."""

    def __init__(self):
        """Initialize the lookup service."""
        self.params_manager = None
        self._initialize_manager()

    def _initialize_manager(self):
        """Initialize the parameters manager with error handling."""
        try:
            self.params_manager = get_optimal_parameters_manager()
            # Test connection
            self.params_manager.get_all_symbols_with_parameters()
            logger.info("Parameter lookup service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize parameter lookup service: {e}")
            self.params_manager = None

    def get_optimal_parameters_for_symbol(
        self, symbol: str, force_fallback: bool = False
    ) -> Dict:
        """
        Get optimal parameters for a symbol with fallbacks.

        Args:
            symbol: Stock symbol
            force_fallback: If True, skip lookup and use defaults

        Returns:
            Dict with parameters and metadata
        """
        try:
            # If service is not available or forced fallback, use defaults
            if force_fallback or self.params_manager is None:
                logger.info(
                    f"Using default parameters for {symbol} (service unavailable or forced)"
                )
                return {
                    **DEFAULT_PROPHET_PARAMS,
                    "symbol": symbol,
                    "source": "default_fallback",
                }

            # Try to get optimized parameters
            logger.debug(f"Looking up optimal parameters for {symbol}")
            optimal_params = self.params_manager.get_parameters(symbol)

            if optimal_params:
                logger.info(f"Found optimal parameters for {symbol}")
                return {
                    "symbol": optimal_params["symbol"],
                    "changepoint_prior_scale": optimal_params[
                        "changepoint_prior_scale"
                    ],
                    "seasonality_prior_scale": optimal_params[
                        "seasonality_prior_scale"
                    ],
                    "rmse": optimal_params["rmse"],
                    "mae": optimal_params["mae"],
                    "smape": optimal_params["smape"],
                    "optimization_date": optimal_params["optimization_date"],
                    "data_points_used": optimal_params["data_points_used"],
                    "source": "optimized",
                }
            else:
                logger.info(f"No optimal parameters found for {symbol}, using defaults")
                return {
                    **DEFAULT_PROPHET_PARAMS,
                    "symbol": symbol,
                    "source": "default_no_optimization",
                }

        except Exception as e:
            logger.error(f"Error looking up parameters for {symbol}: {e}")
            return {
                **DEFAULT_PROPHET_PARAMS,
                "symbol": symbol,
                "source": "default_error",
            }

    def get_symbols_with_optimization(self) -> list:
        """Get list of symbols that have optimal parameters."""
        try:
            if self.params_manager is None:
                return []

            symbols = self.params_manager.get_all_symbols_with_parameters()
            logger.info(f"Found {len(symbols)} symbols with optimal parameters")
            return symbols

        except Exception as e:
            logger.error(f"Error getting symbols with optimization: {e}")
            return []

    def get_optimization_summary(self) -> Optional[pd.DataFrame]:
        """Get optimization summary for all symbols."""
        try:
            if self.params_manager is None:
                return None

            summary = self.params_manager.get_optimization_summary()
            logger.info(f"Retrieved optimization summary for {len(summary)} symbols")
            return summary

        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return None


# Global service instance
_lookup_service = None


def get_lookup_service() -> ParameterLookupService:
    """Get or create the global parameter lookup service."""
    global _lookup_service
    if _lookup_service is None:
        _lookup_service = ParameterLookupService()
    return _lookup_service


def get_optimal_parameters_for_symbol(
    symbol: str, force_fallback: bool = False
) -> Dict:
    """
    Main function to get optimal parameters for a symbol.

    Args:
        symbol: Stock symbol
        force_fallback: If True, skip optimization lookup

    Returns:
        Dict with parameters and metadata including:
        - changepoint_prior_scale: Prophet parameter
        - seasonality_prior_scale: Prophet parameter
        - rmse, mae, smape: Performance metrics (if available)
        - source: 'optimized', 'default_fallback', 'default_no_optimization', or 'default_error'
    """
    service = get_lookup_service()
    return service.get_optimal_parameters_for_symbol(symbol, force_fallback)


def should_use_optimal_parameters(symbol: str) -> Tuple[bool, str]:
    """
    Determine if optimal parameters should be used for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Tuple of (should_use, reason)
    """
    try:
        service = get_lookup_service()
        optimized_symbols = service.get_symbols_with_optimization()

        if symbol in optimized_symbols:
            return True, "Optimal parameters available"
        else:
            return False, "No optimization data available"

    except Exception as e:
        logger.error(f"Error checking optimization status for {symbol}: {e}")
        return False, f"Error: {e}"


def display_parameter_source_info(params: Dict):
    """
    Display information about parameter source in Streamlit UI.

    Args:
        params: Parameters dict from get_optimal_parameters_for_symbol
    """
    symbol = params.get("symbol", "Unknown")
    source = params.get("source", "unknown")

    if source == "optimized":
        st.success(f"DONE: Using optimized parameters for {symbol}")

        # Show optimization details
        with st.expander("Optimization Details", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("RMSE", f"{params.get('rmse', 0):.4f}")
            with col2:
                st.metric("MAE", f"{params.get('mae', 0):.4f}")
            with col3:
                st.metric("SMAPE", f"{params.get('smape', 0):.4f}")

            st.write(
                f"**Optimization Date:** {params.get('optimization_date', 'Unknown')}"
            )
            st.write(
                f"**Data Points Used:** {params.get('data_points_used', 'Unknown'):,}"
            )
            st.write(
                f"**Parameters:** changepoint_prior_scale={params.get('changepoint_prior_scale', 0):.3f}, "
                f"seasonality_prior_scale={params.get('seasonality_prior_scale', 0):.2f}"
            )

    elif source == "default_no_optimization":
        st.info(
            f"Using default parameters for {symbol} (no optimization data available)"
        )
        st.write("TIP: Run the optimization script to improve forecasting accuracy!")

    elif source == "default_fallback":
        st.info(
            f"Using default parameters for {symbol} (optimization service unavailable)"
        )

    elif source == "default_error":
        st.warning(
            f"WARNING: Using default parameters for {symbol} due to lookup error"
        )

    else:
        st.info(f"Using default parameters for {symbol}")


def update_ui_sliders_with_optimal_parameters(
    symbol: str, user_override: bool = False
) -> Tuple[float, float]:
    """
    Update Streamlit UI sliders with optimal parameters.

    Args:
        symbol: Stock symbol
        user_override: If True, allow user to override optimal parameters

    Returns:
        Tuple of (changepoint_prior_scale, seasonality_prior_scale)
    """
    # Get optimal parameters
    params = get_optimal_parameters_for_symbol(symbol)

    # Display parameter source information
    display_parameter_source_info(params)

    # Set default values from optimization or fallback
    default_changepoint = params.get(
        "changepoint_prior_scale", DEFAULT_PROPHET_PARAMS["changepoint_prior_scale"]
    )
    default_seasonality = params.get(
        "seasonality_prior_scale", DEFAULT_PROPHET_PARAMS["seasonality_prior_scale"]
    )

    # Create sliders (users can still override optimal values)
    if user_override or params.get("source") != "optimized":
        # Show sliders for adjustment
        changepoint_prior_scale = st.slider(
            "Changepoint Prior Scale",
            min_value=PARAM_RANGES["changepoint_prior_scale"]["min"],
            max_value=PARAM_RANGES["changepoint_prior_scale"]["max"],
            value=default_changepoint,
            step=PARAM_RANGES["changepoint_prior_scale"]["step"],
            format=PARAM_RANGES["changepoint_prior_scale"]["format"],
            help="Controls trend flexibility. Lower = smoother trends, Higher = more changepoints detected",
        )

        seasonality_prior_scale = st.slider(
            "Seasonality Prior Scale",
            min_value=PARAM_RANGES["seasonality_prior_scale"]["min"],
            max_value=PARAM_RANGES["seasonality_prior_scale"]["max"],
            value=default_seasonality,
            step=PARAM_RANGES["seasonality_prior_scale"]["step"],
            format=PARAM_RANGES["seasonality_prior_scale"]["format"],
            help="Controls seasonal variation strength. Lower = smoother seasonality, Higher = more pronounced patterns",
        )

        # Warn if user changed optimal parameters
        if params.get("source") == "optimized":
            if (
                changepoint_prior_scale != default_changepoint
                or seasonality_prior_scale != default_seasonality
            ):
                st.warning(
                    "WARNING: You've modified the optimal parameters. This may reduce forecasting accuracy."
                )

    else:
        # Use optimal parameters directly, show as info
        st.write(f"**Using Optimal Parameters:**")
        st.write(f"• Changepoint Prior Scale: {default_changepoint:.3f}")
        st.write(f"• Seasonality Prior Scale: {default_seasonality:.2f}")

        changepoint_prior_scale = default_changepoint
        seasonality_prior_scale = default_seasonality

        # Option to override
        if st.checkbox(
            "Override optimal parameters",
            help="Check this to manually adjust the optimal parameters",
        ):
            return update_ui_sliders_with_optimal_parameters(symbol, user_override=True)

    return changepoint_prior_scale, seasonality_prior_scale


def get_optimization_status_summary() -> Dict:
    """
    Get summary of optimization status across all symbols.

    Returns:
        Dict with optimization statistics
    """
    try:
        service = get_lookup_service()
        optimized_symbols = service.get_symbols_with_optimization()
        summary_df = service.get_optimization_summary()

        if summary_df is not None and not summary_df.empty:
            return {
                "total_optimized": len(optimized_symbols),
                "avg_rmse": summary_df["rmse"].mean(),
                "best_rmse": summary_df["rmse"].min(),
                "worst_rmse": summary_df["rmse"].max(),
                "best_symbol": summary_df.loc[summary_df["rmse"].idxmin(), "symbol"],
                "summary_available": True,
            }
        else:
            return {
                "total_optimized": len(optimized_symbols),
                "summary_available": False,
            }

    except Exception as e:
        logger.error(f"Error getting optimization status: {e}")
        return {"total_optimized": 0, "summary_available": False, "error": str(e)}


def display_optimization_status():
    """Display optimization status in Streamlit UI."""
    status = get_optimization_status_summary()

    if status.get("summary_available", False):
        st.subheader("Optimization Status")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Optimized Symbols", status["total_optimized"])
        with col2:
            st.metric("Average RMSE", f"{status['avg_rmse']:.3f}")
        with col3:
            st.metric("Best RMSE", f"{status['best_rmse']:.3f}")
        with col4:
            st.metric("Best Symbol", status["best_symbol"])

        if st.button("Show Full Optimization Report"):
            service = get_lookup_service()
            summary_df = service.get_optimization_summary()
            if summary_df is not None and not summary_df.empty:
                st.dataframe(summary_df)

    elif status["total_optimized"] > 0:
        st.info(f"DONE: {status['total_optimized']} symbols have optimal parameters")

    else:
        st.warning(
            "WARNING: No optimized parameters available. Run optimization script to improve accuracy!"
        )

        if st.button("How to run optimization"):
            st.code(
                """
# Optimize all symbols (may take several hours)
python scripts/optimize_symbol_parameters.py

# Optimize specific symbols  
python scripts/optimize_symbol_parameters.py --symbols AAPL,GOOGL,MSFT

# Test mode (faster, reduced parameter grid)
python scripts/optimize_symbol_parameters.py --test-mode
            """,
                language="bash",
            )


# Convenience function for backwards compatibility
def get_default_parameters() -> Dict:
    """Get default Prophet parameters."""
    return DEFAULT_PROPHET_PARAMS.copy()
