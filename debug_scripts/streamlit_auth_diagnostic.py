#!/usr/bin/env python3
"""
Streamlit Cloud BigQuery Authentication Diagnostic

This script helps diagnose BigQuery authentication issues specifically
for Streamlit Cloud deployments. Run this locally or deploy it to test
authentication methods.

Usage:
    python streamlit_auth_diagnostic.py

Author: Shane
Created: 2025-09-13
"""

import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_streamlit_secrets():
    """Test Streamlit secrets access"""
    try:
        import streamlit as st

        logger.info("✅ Streamlit available")

        if hasattr(st, "secrets"):
            logger.info("✅ st.secrets available")

            # Try to access BigQuery credentials
            try:
                creds = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", None)
                if creds:
                    logger.info(
                        "✅ GOOGLE_APPLICATION_CREDENTIALS_JSON found in secrets"
                    )

                    # Validate JSON format
                    try:
                        cred_info = json.loads(creds)
                        required_fields = [
                            "type",
                            "project_id",
                            "private_key",
                            "client_email",
                        ]
                        missing_fields = [
                            field for field in required_fields if field not in cred_info
                        ]

                        if missing_fields:
                            logger.error(
                                f"❌ Missing required fields in service account JSON: {missing_fields}"
                            )
                            return False
                        else:
                            logger.info("✅ Service account JSON format valid")
                            logger.info(f"✅ Project ID: {cred_info.get('project_id')}")
                            logger.info(
                                f"✅ Service Account: {cred_info.get('client_email')}"
                            )
                            return True

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid JSON format in secrets: {e}")
                        return False
                else:
                    logger.warning(
                        "⚠️  GOOGLE_APPLICATION_CREDENTIALS_JSON not found in secrets"
                    )
                    return False

            except Exception as e:
                logger.error(f"❌ Error accessing secrets: {e}")
                return False
        else:
            logger.warning(
                "⚠️  st.secrets not available (not running in Streamlit context)"
            )
            return False

    except ImportError:
        logger.warning("⚠️  Streamlit not available")
        return False


def test_environment_variable():
    """Test environment variable authentication"""
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

    if creds_json:
        logger.info("✅ GOOGLE_APPLICATION_CREDENTIALS_JSON found in environment")

        try:
            cred_info = json.loads(creds_json)
            logger.info("✅ Environment variable JSON format valid")
            logger.info(f"✅ Project ID: {cred_info.get('project_id')}")
            logger.info(f"✅ Service Account: {cred_info.get('client_email')}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON format in environment variable: {e}")
            return False
    else:
        logger.warning(
            "⚠️  GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment"
        )
        return False


def test_bigquery_connection():
    """Test BigQuery connection with timeout"""
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        import streamlit as st

        logger.info("Testing BigQuery connection...")

        # Try Streamlit secrets first
        service_account_key = None

        if hasattr(st, "secrets"):
            try:
                service_account_key = st.secrets.get(
                    "GOOGLE_APPLICATION_CREDENTIALS_JSON", None
                )
                if service_account_key:
                    logger.info("Using credentials from Streamlit secrets")
            except:
                pass

        # Fallback to environment variable
        if not service_account_key:
            service_account_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            if service_account_key:
                logger.info("Using credentials from environment variable")

        if not service_account_key:
            logger.error("❌ No service account credentials found")
            return False

        # Create credentials and client
        credentials_info = json.loads(service_account_key)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info
        )
        project_id = credentials_info.get("project_id")

        client = bigquery.Client(credentials=credentials, project=project_id)

        # Test connection with timeout
        query = "SELECT 1 as test_value LIMIT 1"
        job_config = bigquery.QueryJobConfig()
        job_config.use_query_cache = True

        query_job = client.query(query, job_config=job_config, timeout=10)
        results = query_job.result(timeout=10)
        list(results)  # Consume results

        logger.info("✅ BigQuery connection test successful")
        return True

    except Exception as e:
        logger.error(f"❌ BigQuery connection test failed: {e}")
        return False


def test_dataset_access():
    """Test specific dataset access"""
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        import streamlit as st

        # Get credentials (same logic as above)
        service_account_key = None

        if hasattr(st, "secrets"):
            try:
                service_account_key = st.secrets.get(
                    "GOOGLE_APPLICATION_CREDENTIALS_JSON", None
                )
            except:
                pass

        if not service_account_key:
            service_account_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

        if not service_account_key:
            logger.error("❌ No credentials for dataset access test")
            return False

        credentials_info = json.loads(service_account_key)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info
        )
        project_id = credentials_info.get("project_id", "stock-forecasting-tool-2025")

        client = bigquery.Client(credentials=credentials, project=project_id)

        # Test dataset access
        dataset_id = "stock_data"
        table_id = "raw_stock_data"

        # Check dataset exists
        dataset_ref = client.dataset(dataset_id)
        dataset = client.get_dataset(dataset_ref)
        logger.info(f"✅ Dataset '{dataset_id}' accessible")

        # Check table exists
        table_ref = dataset_ref.table(table_id)
        table = client.get_table(table_ref)
        logger.info(f"✅ Table '{table_id}' accessible")

        # Test query on actual data
        query = f"SELECT symbol, COUNT(*) as row_count FROM `{project_id}.{dataset_id}.{table_id}` GROUP BY symbol LIMIT 5"
        query_job = client.query(query, timeout=15)
        results = list(query_job.result(timeout=15))

        logger.info(f"✅ Data access successful - found {len(results)} symbols")
        for row in results[:3]:  # Show first 3 symbols
            logger.info(f"  📈 {row.symbol}: {row.row_count} rows")

        return True

    except Exception as e:
        logger.error(f"❌ Dataset access test failed: {e}")
        return False


def main():
    """Run all diagnostic tests"""
    logger.info("=" * 60)
    logger.info("STREAMLIT CLOUD BIGQUERY AUTHENTICATION DIAGNOSTIC")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    results = {}

    # Test 1: Streamlit Secrets
    logger.info("\n🔍 Test 1: Streamlit Secrets Access")
    logger.info("-" * 40)
    results["streamlit_secrets"] = test_streamlit_secrets()

    # Test 2: Environment Variables
    logger.info("\n🔍 Test 2: Environment Variable Access")
    logger.info("-" * 40)
    results["environment_vars"] = test_environment_variable()

    # Test 3: BigQuery Connection
    logger.info("\n🔍 Test 3: BigQuery Connection Test")
    logger.info("-" * 40)
    results["bigquery_connection"] = test_bigquery_connection()

    # Test 4: Dataset Access
    logger.info("\n🔍 Test 4: Dataset and Data Access")
    logger.info("-" * 40)
    results["dataset_access"] = test_dataset_access()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    passed_tests = sum(results.values())
    total_tests = len(results)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info(
            "🎉 All tests passed! BigQuery authentication is working correctly."
        )
    else:
        logger.info("⚠️  Some tests failed. Check the errors above for details.")
        logger.info(
            "💡 Refer to STREAMLIT_CLOUD_DEPLOYMENT.md for troubleshooting steps."
        )


if __name__ == "__main__":
    main()
