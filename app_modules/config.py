# app_modules/config.py
import os
from dotenv import load_dotenv

def load_environment_variables():
    """
    Loads environment variables from a .env file and returns API keys.
    """
    load_dotenv()
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
    finnhub_api_key = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")
    return alpha_vantage_key, finnhub_api_key

# Application Constants
EXCHANGE_CODE = "US"