#!/usr/bin/env python
"""
Test script to verify BigQuery connection fix
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print(f"🔧 GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT')}")

# Test the BigQuery connection
from app_modules.bigquery_client import get_bigquery_client

# Initialize client
print("📡 Initializing BigQuery client...")
client = get_bigquery_client()

# Test connection
print("🔗 Testing BigQuery connection...")
if client.test_connection():
    print('✅ BigQuery connection successful!')
    
    # Get available symbols
    print("📊 Getting available symbols...")
    symbols = client.get_available_symbols()
    print(f'📊 Found {len(symbols)} symbols in BigQuery')
    print(f'🔍 Sample symbols: {symbols[:5]}')
    
    # Test data query for AAPL
    if 'AAPL' in symbols:
        print("📈 Testing data retrieval for AAPL...")
        data = client.query_stock_data('AAPL', start_date='2025-08-01')
        print(f'📈 Retrieved {len(data)} rows for AAPL since August 2025')
        if not data.empty:
            print(f'💰 Latest AAPL close: ${data["close"].iloc[-1]:.2f}')
            print(f'📅 Data range: {data.index.min()} to {data.index.max()}')
    else:
        print("⚠️  AAPL not found in available symbols")
        
else:
    print('❌ BigQuery connection failed')

print("✨ BigQuery configuration test complete!")
