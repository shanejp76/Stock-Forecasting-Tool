"""
Tests for data handling functionality.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add the app_modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app_modules'))

try:
    from app_modules.data_handler import process_technical_indicators
    from app_modules.config import load_environment_variables
except ImportError:
    # Skip tests if modules can't be imported
    pytest.skip("App modules not available", allow_module_level=True)


class TestDataHandler:
    """Test data handling functions."""
    
    def test_process_technical_indicators_basic(self, sample_stock_data):
        """Test basic technical indicators processing."""
        # Test that the function runs without error
        result = process_technical_indicators(sample_stock_data.copy(), 'Close')
        
        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Check that original columns are preserved
        for col in sample_stock_data.columns:
            assert col in result.columns
    
    def test_process_technical_indicators_empty_data(self):
        """Test technical indicators with empty data."""
        empty_df = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        try:
            result = process_technical_indicators(empty_df, 'Close')
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # If it raises an exception, it should be a specific type
            assert isinstance(e, (KeyError, ValueError))
    
    def test_process_technical_indicators_missing_column(self, sample_stock_data):
        """Test technical indicators with missing price column."""
        # Should handle missing column gracefully
        try:
            result = process_technical_indicators(sample_stock_data.copy(), 'NonexistentColumn')
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Expected to raise KeyError for missing column
            assert isinstance(e, KeyError)


class TestConfig:
    """Test configuration functions."""
    
    def test_load_environment_variables_with_mock(self, mock_environment_variables):
        """Test loading environment variables."""
        av_key, finnhub_key = load_environment_variables()
        
        assert av_key == "test_av_key"
        assert finnhub_key == "test_finnhub_key"
    
    def test_load_environment_variables_missing(self, monkeypatch):
        """Test loading environment variables when they're missing."""
        # Remove environment variables
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        
        av_key, finnhub_key = load_environment_variables()
        
        # Should return None for missing variables
        assert av_key is None
        assert finnhub_key is None


class TestDataValidation:
    """Test data validation functions."""
    
    def test_sample_data_structure(self, sample_stock_data):
        """Test that sample data has expected structure."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Volume']
        
        for col in required_columns:
            assert col in sample_stock_data.columns
        
        # Check data types
        assert sample_stock_data['Volume'].dtype in ['int64', 'float64']
        assert len(sample_stock_data) > 0
    
    def test_data_integrity(self, sample_stock_data):
        """Test basic data integrity rules."""
        # High should be >= Open, Close, Low
        assert (sample_stock_data['High'] >= sample_stock_data['Open']).all()
        assert (sample_stock_data['High'] >= sample_stock_data['Close']).all()
        assert (sample_stock_data['High'] >= sample_stock_data['Low']).all()
        
        # Low should be <= Open, Close, High
        assert (sample_stock_data['Low'] <= sample_stock_data['Open']).all()
        assert (sample_stock_data['Low'] <= sample_stock_data['Close']).all()
        assert (sample_stock_data['Low'] <= sample_stock_data['High']).all()
        
        # Volume should be positive
        assert (sample_stock_data['Volume'] > 0).all()
