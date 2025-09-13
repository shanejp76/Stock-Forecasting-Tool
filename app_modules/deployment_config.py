"""
Deployment Configuration Module

This module handles authentication and service configuration for different deployment environments.
It provides fallback mechanisms when BigQuery is unavailable in deployed environments.

Author: Shane
Created: 2025-09-10
"""

import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DeploymentConfig:
    """Configuration manager for different deployment environments"""

    def __init__(self):
        self.environment = self._detect_environment()
        self.bigquery_available = False
        self._check_bigquery_availability()

    def _detect_environment(self) -> str:
        """Detect the current deployment environment"""
        if os.getenv("STREAMLIT_CLOUD"):
            return "streamlit_cloud"
        elif os.getenv("GOOGLE_CLOUD_PROJECT"):
            return "google_cloud"
        elif os.getenv("HEROKU"):
            return "heroku"
        else:
            return "local"

    def _check_bigquery_availability(self) -> None:
        """Check if BigQuery authentication is properly configured"""
        try:
            # Priority 1: Check Streamlit secrets (most reliable for Streamlit Cloud)
            try:
                import streamlit as st

                if hasattr(st, "secrets") and st.secrets.get(
                    "GOOGLE_APPLICATION_CREDENTIALS_JSON"
                ):
                    self.bigquery_available = True
                    logger.info(
                        "BigQuery service account credentials found in Streamlit secrets"
                    )
                    return
            except Exception:
                pass  # Streamlit not available or secrets not accessible

            # Priority 2: Check for service account key in environment
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
                self.bigquery_available = True
                logger.info("BigQuery service account credentials found in environment")
                return

            # Priority 3: Check for default credentials file
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                self.bigquery_available = True
                logger.info("BigQuery credentials file found")
                return

            # Priority 4: For local development, try default auth with timeout
            if self.environment == "local":
                try:
                    from google.auth import default
                    import google.auth.transport.requests

                    # Use short timeout to prevent hanging
                    request = google.auth.transport.requests.Request(timeout=3)
                    default(request=request)
                    self.bigquery_available = True
                    logger.info("BigQuery default authentication available")
                except Exception as e:
                    logger.warning(
                        f"BigQuery default authentication not available: {e}"
                    )

        except Exception as e:
            logger.warning(f"BigQuery availability check failed: {e}")

    def get_data_source_config(self) -> Dict[str, Any]:
        """Get configuration for data sources based on environment"""
        config = {
            "bigquery_available": self.bigquery_available,
            "environment": self.environment,
            "fallback_to_alpha_vantage": not self.bigquery_available,
        }

        if not self.bigquery_available:
            logger.info(f"Running in {self.environment} environment without BigQuery")
            logger.info("Application will use Alpha Vantage API only")

        return config

    def should_show_bigquery_option(self) -> bool:
        """Determine if BigQuery option should be shown in UI"""
        return self.bigquery_available

    def get_default_data_source(self) -> str:
        """Get the default data source for the current environment"""
        return "bigquery" if self.bigquery_available else "alpha_vantage"


# Global deployment config instance
deployment_config = DeploymentConfig()
