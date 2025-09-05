"""Quick test to debug chart date column issue"""
import pandas as pd
import sys
import os

# Add the current directory to Python path
sys.path.append(os.getcwd())

from app_modules.data_pipeline import merge_forecast_with_data

# Create simple test data
test_forecast = pd.DataFrame({
    'ds': pd.date_range('2024-01-01', periods=5),
    'yhat': [100, 101, 102, 103, 104],
    'yhat_lower': [95, 96, 97, 98, 99],
    'yhat_upper': [105, 106, 107, 108, 109]
})

test_data = pd.DataFrame({
    'Date': pd.date_range('2023-12-28', periods=5),
    'Close': [98, 99, 100, 101, 102]
})

print("Original forecast columns:", test_forecast.columns.tolist())
print("Original data columns:", test_data.columns.tolist())

try:
    result = merge_forecast_with_data(test_forecast, test_data, 30)
    print("SUCCESS! Merged forecast columns:", result.columns.tolist())
    print("Sample rows:")
    print(result.head())
except Exception as e:
    print("ERROR:", str(e))
    import traceback
    traceback.print_exc()
