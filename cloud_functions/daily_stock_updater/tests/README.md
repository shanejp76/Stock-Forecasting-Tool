# Testing Utilities

This directory contains testing scripts for the Cloud Function.

## Files

- **test_local.py** - Local testing without mocking (requires real API keys)
- **test_mocked.py** - Mocked testing for offline development
- **test_trading_days.py** - Specific tests for trading day detection logic

## Usage

```bash
# Run mocked tests (recommended for development)
python tests/test_mocked.py

# Run with real API (requires valid Alpha Vantage key)
python tests/test_local.py

# Test trading day logic
python tests/test_trading_days.py
```

## Note

These are development testing utilities. For production testing, use the Cloud Function's built-in logging and monitoring capabilities.
