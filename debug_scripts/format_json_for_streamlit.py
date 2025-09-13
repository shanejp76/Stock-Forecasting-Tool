#!/usr/bin/env python3
"""
Service Account JSON Formatter for Streamlit Cloud

This script helps format your service account JSON properly for Streamlit Cloud
secrets to avoid "Invalid control character" errors.

Usage:
    python format_json_for_streamlit.py

Author: Shane
Created: 2025-09-13
"""

import json
import re


def format_json_for_streamlit():
    """Format service account JSON for Streamlit Cloud secrets"""

    try:
        # Try to load from local Streamlit secrets first
        import streamlit as st

        if hasattr(st, "secrets"):
            json_str = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if json_str:
                print("✅ Found JSON in local Streamlit secrets")
                # Parse and re-serialize to ensure proper formatting
                credentials_info = json.loads(json_str)

                # Clean format for Streamlit Cloud
                clean_json = json.dumps(credentials_info, separators=(",", ":"))

                # Escape any remaining control characters
                clean_json = clean_json.replace("\\n", "\\\\n")
                clean_json = clean_json.replace("\\r", "\\\\r")
                clean_json = clean_json.replace("\\t", "\\\\t")

                print("\n" + "=" * 60)
                print("FORMATTED JSON FOR STREAMLIT CLOUD SECRETS:")
                print("=" * 60)
                print("\nCopy this EXACT text to your Streamlit Cloud secrets:")
                print(
                    '\nGOOGLE_APPLICATION_CREDENTIALS_JSON = """' + clean_json + '"""'
                )
                print("\n" + "=" * 60)

                # Validate the formatted JSON
                try:
                    test_parse = json.loads(clean_json)
                    print("✅ Formatted JSON is valid")
                    print(f"✅ Project ID: {test_parse.get('project_id', 'NOT FOUND')}")
                    print(
                        f"✅ Service Account: {test_parse.get('client_email', 'NOT FOUND')}"
                    )
                except json.JSONDecodeError as e:
                    print(f"❌ Formatted JSON is invalid: {e}")

                return clean_json

    except Exception as e:
        print(f"❌ Error accessing local secrets: {e}")

    # Fallback: Load from credentials.json file
    try:
        with open("credentials.json", "r") as f:
            credentials_info = json.load(f)

        print("✅ Found credentials.json file")

        # Clean format for Streamlit Cloud
        clean_json = json.dumps(credentials_info, separators=(",", ":"))

        # Escape control characters properly
        clean_json = clean_json.replace("\\n", "\\\\n")
        clean_json = clean_json.replace("\\r", "\\\\r")
        clean_json = clean_json.replace("\\t", "\\\\t")

        print("\n" + "=" * 60)
        print("FORMATTED JSON FOR STREAMLIT CLOUD SECRETS:")
        print("=" * 60)
        print("\nCopy this EXACT text to your Streamlit Cloud secrets:")
        print('\nGOOGLE_APPLICATION_CREDENTIALS_JSON = """' + clean_json + '"""')
        print("\n" + "=" * 60)

        # Validate the formatted JSON
        try:
            test_parse = json.loads(clean_json)
            print("✅ Formatted JSON is valid")
            print(f"✅ Project ID: {test_parse.get('project_id', 'NOT FOUND')}")
            print(f"✅ Service Account: {test_parse.get('client_email', 'NOT FOUND')}")
        except json.JSONDecodeError as e:
            print(f"❌ Formatted JSON is invalid: {e}")

        return clean_json

    except FileNotFoundError:
        print("❌ credentials.json file not found")
        print("\nPlease ensure you have either:")
        print("1. A working .streamlit/secrets.toml file, or")
        print("2. A credentials.json file in the current directory")

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in credentials.json: {e}")

    return None


if __name__ == "__main__":
    print("🔧 SERVICE ACCOUNT JSON FORMATTER")
    print("=" * 60)
    print("This script will format your service account JSON")
    print("for proper use in Streamlit Cloud secrets.")
    print("=" * 60 + "\n")

    formatted_json = format_json_for_streamlit()

    if formatted_json:
        print("\n📋 NEXT STEPS:")
        print("1. Copy the formatted JSON above")
        print("2. Go to your Streamlit Cloud app settings")
        print("3. Navigate to 'Secrets' section")
        print("4. Replace the current GOOGLE_APPLICATION_CREDENTIALS_JSON value")
        print("5. Save and restart your app")
        print("\n⚠️  IMPORTANT: Use the EXACT formatting shown above")
    else:
        print("\n❌ Could not format JSON. Please check your credentials setup.")
