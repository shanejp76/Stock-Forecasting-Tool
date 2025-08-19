"""
Test configuration and fixtures for the Stock Forecasting Tool.
"""
import os
import pytest
import sys
from unittest.mock import Mock

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test if dependencies are available
DEPENDENCIES_AVAILABLE = True
try:
    import pandas as pd
    import numpy as np
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing."""
    if not DEPENDENCIES_AVAILABLE:
        pytest.skip("Dependencies not available")
    
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible tests
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': 100 + pd.Series(range(100)) * 0.1,
        'High': 102 + pd.Series(range(100)) * 0.1,
        'Low': 98 + pd.Series(range(100)) * 0.1,
        'Close': 101 + pd.Series(range(100)) * 0.1,
        'Adjusted Close': 101 + pd.Series(range(100)) * 0.1,
        'Volume': 1000000 + pd.Series(range(100)) * 1000
    })
    return data.set_index('Date')


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle missing dependencies."""
    if not DEPENDENCIES_AVAILABLE:
        skip_deps = pytest.mark.skip(reason="Required dependencies not available")
        for item in items:
            item.add_marker(skip_deps)


@pytest.fixture
def mock_api_response():
    """Mock API response for testing."""
    return {
        'Meta Data': {
            '1. Information': 'Daily Prices (open, high, low, close) and Volumes',
            '2. Symbol': 'AAPL',
            '3. Last Refreshed': '2024-01-01',
            '4. Output Size': 'Compact',
            '5. Time Zone': 'US/Eastern'
        },
        'Time Series (Daily)': {
            '2024-01-01': {
                '1. open': '100.0',
                '2. high': '102.0',
                '3. low': '98.0',
                '4. close': '101.0',
                '5. adjusted close': '101.0',
                '6. volume': '1000000'
            }
        }
    }


@pytest.fixture
def mock_environment_variables(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_av_key")
    monkeypatch.setenv("FINNHUB_API_KEY", "test_finnhub_key")


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for testing."""
    mock_st = Mock()
    mock_st.error = Mock()
    mock_st.stop = Mock()
    mock_st.spinner = Mock()
    mock_st.expander = Mock()
    mock_st.sidebar = Mock()
    return mock_st
