#!/usr/bin/env python3
"""
GitHub Secrets JSON Formatter for GCP Service Account

This script formats your service account JSON for GitHub Actions secrets.
Unlike Streamlit Cloud, GitHub secrets don't need escaped characters.

Usage:
    python format_github_secrets.py

Author: Shane
Created: 2025-09-13
"""

import json
import os


def format_for_github_secrets():
    """Format service account JSON for GitHub Actions secrets"""

    try:
        # Try to load from local Streamlit secrets first
        import streamlit as st

        if hasattr(st, "secrets"):
            json_str = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if json_str:
                print("✅ Found JSON in local Streamlit secrets")
                # Parse and re-serialize without escaping
                credentials_info = json.loads(json_str)

                # GitHub secrets format (compact, no escaping needed)
                github_json = json.dumps(credentials_info, separators=(",", ":"))

                print("\n" + "=" * 80)
                print("GITHUB SECRETS CONFIGURATION")
                print("=" * 80)
                print("\n1. Go to your GitHub repository")
                print("2. Navigate to: Settings → Secrets and variables → Actions")
                print("3. Click 'New repository secret'")
                print("4. Add these secrets:")
                print("\n" + "-" * 80)

                print("\nSecret Name: GCP_PROJECT_ID")
                print(
                    f"Secret Value: {credentials_info.get('project_id', 'stock-forecasting-tool-2025')}"
                )

                print("\nSecret Name: GCP_REGION")
                print("Secret Value: us-central1")

                print("\nSecret Name: GCP_SERVICE_ACCOUNT_KEY")
                print("Secret Value:")
                print("-" * 40)
                print(github_json)
                print("-" * 40)

                print("\n" + "=" * 80)
                print("IMPORTANT NOTES:")
                print("=" * 80)
                print("• Copy the entire JSON (including curly braces)")
                print(
                    "• Paste directly into GitHub secrets - NO QUOTES needed around it"
                )
                print("• GitHub handles the JSON parsing automatically")
                print("• Do NOT escape quotes or newlines for GitHub secrets")
                print("• Make sure there are no trailing spaces")

                # Validate the JSON
                try:
                    test_parse = json.loads(github_json)
                    print(f"\n✅ JSON is valid")
                    print(f"✅ Project ID: {test_parse.get('project_id', 'NOT FOUND')}")
                    print(
                        f"✅ Service Account: {test_parse.get('client_email', 'NOT FOUND')}"
                    )
                except json.JSONDecodeError as e:
                    print(f"\n❌ JSON validation failed: {e}")

                return github_json

    except Exception as e:
        print(f"❌ Error accessing local secrets: {e}")

    # Fallback: Load from credentials.json file
    try:
        with open("credentials.json", "r") as f:
            credentials_info = json.load(f)

        print("✅ Found credentials.json file")

        # GitHub secrets format (compact, no escaping needed)
        github_json = json.dumps(credentials_info, separators=(",", ":"))

        print("\n" + "=" * 80)
        print("GITHUB SECRETS CONFIGURATION")
        print("=" * 80)
        print("\n1. Go to your GitHub repository")
        print("2. Navigate to: Settings → Secrets and variables → Actions")
        print("3. Click 'New repository secret'")
        print("4. Add these secrets:")
        print("\n" + "-" * 80)

        print("\nSecret Name: GCP_PROJECT_ID")
        print(
            f"Secret Value: {credentials_info.get('project_id', 'stock-forecasting-tool-2025')}"
        )

        print("\nSecret Name: GCP_REGION")
        print("Secret Value: us-central1")

        print("\nSecret Name: GCP_SERVICE_ACCOUNT_KEY")
        print("Secret Value:")
        print("-" * 40)
        print(github_json)
        print("-" * 40)

        print("\n" + "=" * 80)
        print("IMPORTANT NOTES:")
        print("=" * 80)
        print("• Copy the entire JSON (including curly braces)")
        print("• Paste directly into GitHub secrets - NO QUOTES needed around it")
        print("• GitHub handles the JSON parsing automatically")
        print("• Do NOT escape quotes or newlines for GitHub secrets")
        print("• Make sure there are no trailing spaces")

        # Validate the JSON
        try:
            test_parse = json.loads(github_json)
            print(f"\n✅ JSON is valid")
            print(f"✅ Project ID: {test_parse.get('project_id', 'NOT FOUND')}")
            print(f"✅ Service Account: {test_parse.get('client_email', 'NOT FOUND')}")
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON validation failed: {e}")

        return github_json

    except FileNotFoundError:
        print("❌ credentials.json file not found")
        print("\nPlease ensure you have either:")
        print("1. A working .streamlit/secrets.toml file, or")
        print("2. A credentials.json file in the current directory")

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in credentials.json: {e}")

    return None


if __name__ == "__main__":
    print("🔧 GITHUB SECRETS FORMATTER FOR GCP SERVICE ACCOUNT")
    print("=" * 80)
    print("This script formats your service account JSON for GitHub Actions.")
    print("=" * 80 + "\n")

    formatted_json = format_for_github_secrets()

    if not formatted_json:
        print("\n❌ Could not format JSON. Please check your credentials setup.")
