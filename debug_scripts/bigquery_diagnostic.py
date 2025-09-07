#!/usr/bin/env python3
"""
BigQuery Diagnostic Tool with Comprehensive Logging

This script provides comprehensive diagnostics for BigQuery data warehouse issues
with detailed logging capabilities. Use this tool whenever you encounter BigQuery
data corruption, column naming issues, or application integration problems.

Features:
- Comprehensive duplicate detection and analysis
- Detailed logging to timestamped files
- Schema validation and comparison
- Data quality auditing
- Application integration testing
- Symbol coverage analysis

Usage:
    python debug_scripts/bigquery_diagnostic.py
    python debug_scripts/bigquery_diagnostic.py --symbol AAPL
    python debug_scripts/bigquery_diagnostic.py --full-audit
    python debug_scripts/bigquery_diagnostic.py --compare-sources AAPL
    python debug_scripts/bigquery_diagnostic.py --log-level DEBUG
"""

import argparse
import sys
import os
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from app_modules.bigquery_client import BigQueryClient
    from app_modules.data_handler import load_bigquery_data
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Make sure you've activated the environment with activate_env.bat or activate_env.ps1"
    )
    sys.exit(1)


class BigQueryDiagnosticLogger:
    """Enhanced logging system for BigQuery diagnostics."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize logging system with timestamped files."""
        self.log_dir = os.path.join(os.path.dirname(__file__), "logging")
        os.makedirs(self.log_dir, exist_ok=True)

        # Create timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            self.log_dir, f"bigquery_diagnostic_{timestamp}.log"
        )

        # Configure logging
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

        self.logger = logging.getLogger("BigQueryDiagnostic")
        self.logger.info(f"Diagnostic session started - Log file: {self.log_file}")

    def log_section(self, section_name: str):
        """Log section headers for better organization."""
        separator = "=" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"🔍 {section_name}")
        self.logger.info(separator)

    def log_finding(self, finding_type: str, message: str, data: Optional[Dict] = None):
        """Log findings with structured data."""
        emoji = (
            "✅"
            if finding_type == "SUCCESS"
            else "❌" if finding_type == "ERROR" else "⚠️"
        )
        self.logger.info(f"{emoji} {finding_type}: {message}")

        if data:
            self.logger.debug(
                f"Associated data: {json.dumps(data, indent=2, default=str)}"
            )


class BigQueryDiagnostic:
    """Comprehensive BigQuery diagnostic and troubleshooting tool with enhanced logging."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize diagnostic tool with BigQuery client and logging."""
        self.logger_system = BigQueryDiagnosticLogger(log_level)
        self.logger = self.logger_system.logger

        try:
            self.bq_client = BigQueryClient()
            self.raw_client = bigquery.Client()
            self.dataset_id = "stock_data"
            self.table_id = "raw_stock_data"

            self.logger.info(
                f"Connected to BigQuery project: {self.raw_client.project}"
            )
            self.logger_system.log_finding(
                "SUCCESS",
                f"BigQuery connection established",
                {
                    "project": self.raw_client.project,
                    "dataset": self.dataset_id,
                    "table": self.table_id,
                },
            )
        except Exception as e:
            self.logger_system.log_finding(
                "ERROR", f"Failed to connect to BigQuery: {e}"
            )
            sys.exit(1)

    def check_connection_health(self) -> bool:
        """Test basic BigQuery connection and permissions."""
        self.logger_system.log_section("CHECKING BIGQUERY CONNECTION HEALTH")

        try:
            # Test basic query
            query = "SELECT 1 as test_value"
            result = list(self.raw_client.query(query))
            self.logger_system.log_finding("SUCCESS", "Basic query execution")

            # Test dataset access
            dataset_ref = self.raw_client.dataset(self.dataset_id)
            dataset = self.raw_client.get_dataset(dataset_ref)
            self.logger_system.log_finding(
                "SUCCESS", f"Dataset access '{self.dataset_id}'"
            )

            # Test table access
            table_ref = dataset_ref.table(self.table_id)
            table = self.raw_client.get_table(table_ref)
            self.logger_system.log_finding("SUCCESS", f"Table access '{self.table_id}'")

            return True

        except NotFound as e:
            self.logger_system.log_finding("ERROR", f"Resource not found: {e}")
            return False
        except Exception as e:
            self.logger_system.log_finding("ERROR", f"Connection test failed: {e}")
            return False

    def examine_table_schema(self) -> Dict:
        """Examine and report on table schema structure."""
        self.logger_system.log_section("EXAMINING TABLE SCHEMA")

        try:
            table_ref = self.raw_client.dataset(self.dataset_id).table(self.table_id)
            table = self.raw_client.get_table(table_ref)

            schema_info = {
                "columns": [],
                "total_columns": len(table.schema),
                "table_size_mb": (
                    table.num_bytes / (1024 * 1024) if table.num_bytes else 0
                ),
                "total_rows": table.num_rows,
                "created": str(table.created),
                "modified": str(table.modified),
            }

            self.logger.info(f"Table: {table.full_table_id}")
            self.logger.info(f"Total Rows: {table.num_rows:,}")
            self.logger.info(f"Size: {schema_info['table_size_mb']:.2f} MB")
            self.logger.info(f"Created: {table.created}")
            self.logger.info(f"Modified: {table.modified}")
            self.logger.info("\nColumns:")

            for field in table.schema:
                column_info = {
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                    "description": field.description or "No description",
                }
                schema_info["columns"].append(column_info)

                self.logger.info(
                    f"  - {field.name:20} {field.field_type:12} {field.mode:10} | {column_info['description']}"
                )

            self.logger_system.log_finding(
                "SUCCESS", "Schema examination completed", schema_info
            )
            return schema_info

        except Exception as e:
            self.logger_system.log_finding("ERROR", f"Schema examination failed: {e}")
            return {}

    def sample_raw_data(
        self, symbol: str = "AAPL", limit: int = 10
    ) -> Optional[pd.DataFrame]:
        """Sample raw data from BigQuery to examine actual content."""
        self.logger_system.log_section(f"SAMPLING RAW DATA FOR {symbol}")

        try:
            query = f"""
            SELECT *
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = '{symbol}'
            ORDER BY date DESC
            LIMIT {limit}
            """

            df = self.raw_client.query(query).to_dataframe()

            if df.empty:
                self.logger_system.log_finding(
                    "ERROR", f"No data found for symbol: {symbol}"
                )
                return None

            self.logger.info(f"Found {len(df)} rows for {symbol}")
            self.logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            self.logger.info("\nColumn names and types:")

            sample_data = {}
            for col in df.columns:
                dtype = df[col].dtype
                sample_value = df[col].iloc[0] if not df[col].isna().all() else "NULL"
                sample_data[col] = {
                    "dtype": str(dtype),
                    "sample_value": str(sample_value),
                }
                self.logger.info(
                    f"  - {col:20} {str(dtype):15} | Sample: {sample_value}"
                )

            self.logger.info("\nFirst 3 rows:")
            self.logger.info(df.head(3).to_string(index=False))

            self.logger_system.log_finding(
                "SUCCESS",
                f"Data sampling completed for {symbol}",
                {
                    "rows_found": len(df),
                    "date_range": {
                        "min": str(df["date"].min()),
                        "max": str(df["date"].max()),
                    },
                    "columns": sample_data,
                },
            )

            return df

        except Exception as e:
            self.logger_system.log_finding("ERROR", f"Data sampling failed: {e}")
            return None

    def check_comprehensive_duplicates(self, symbol: str = "AAPL") -> Dict:
        """Comprehensive duplicate analysis with detailed logging."""
        self.logger_system.log_section(f"COMPREHENSIVE DUPLICATE ANALYSIS FOR {symbol}")

        duplicate_report = {
            "date_duplicates": 0,
            "exact_duplicates": 0,
            "symbol_date_duplicates": 0,
            "partial_duplicates": 0,
            "duplicate_details": [],
        }

        try:
            # 1. Check for date duplicates within symbol
            date_dup_query = f"""
            SELECT date, COUNT(*) as count
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = '{symbol}'
            GROUP BY date
            HAVING COUNT(*) > 1
            ORDER BY date DESC
            """

            date_dup_result = self.raw_client.query(date_dup_query).to_dataframe()
            duplicate_report["date_duplicates"] = len(date_dup_result)

            if duplicate_report["date_duplicates"] > 0:
                self.logger_system.log_finding(
                    "ERROR",
                    f"Found {duplicate_report['date_duplicates']} duplicate dates for {symbol}",
                )
                self.logger.info("Duplicate dates:")
                self.logger.info(date_dup_result.head().to_string(index=False))

                duplicate_report["duplicate_details"].append(
                    {
                        "type": "date_duplicates",
                        "count": duplicate_report["date_duplicates"],
                        "sample": date_dup_result.head().to_dict("records"),
                    }
                )
            else:
                self.logger_system.log_finding(
                    "SUCCESS", f"No duplicate dates found for {symbol}"
                )

            # 2. Check for exact row duplicates
            exact_dup_query = f"""
            SELECT date, symbol, open, high, low, close, volume, COUNT(*) as count
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = '{symbol}'
            GROUP BY date, symbol, open, high, low, close, volume
            HAVING COUNT(*) > 1
            ORDER BY date DESC
            """

            exact_dup_result = self.raw_client.query(exact_dup_query).to_dataframe()
            duplicate_report["exact_duplicates"] = len(exact_dup_result)

            if duplicate_report["exact_duplicates"] > 0:
                self.logger_system.log_finding(
                    "ERROR",
                    f"Found {duplicate_report['exact_duplicates']} exact duplicate rows for {symbol}",
                )
                duplicate_report["duplicate_details"].append(
                    {
                        "type": "exact_duplicates",
                        "count": duplicate_report["exact_duplicates"],
                        "sample": exact_dup_result.head().to_dict("records"),
                    }
                )
            else:
                self.logger_system.log_finding(
                    "SUCCESS", f"No exact duplicate rows found for {symbol}"
                )

            # 3. Check for symbol-date combinations across entire table
            symbol_date_dup_query = f"""
            SELECT symbol, date, COUNT(*) as count
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            GROUP BY symbol, date
            HAVING COUNT(*) > 1
            ORDER BY symbol, date DESC
            LIMIT 100
            """

            symbol_date_dup_result = self.raw_client.query(
                symbol_date_dup_query
            ).to_dataframe()
            duplicate_report["symbol_date_duplicates"] = len(symbol_date_dup_result)

            if duplicate_report["symbol_date_duplicates"] > 0:
                self.logger_system.log_finding(
                    "ERROR",
                    f"Found {duplicate_report['symbol_date_duplicates']} symbol-date duplicates across warehouse",
                )
                self.logger.info("Sample symbol-date duplicates:")
                self.logger.info(symbol_date_dup_result.head().to_string(index=False))

                duplicate_report["duplicate_details"].append(
                    {
                        "type": "symbol_date_duplicates",
                        "count": duplicate_report["symbol_date_duplicates"],
                        "sample": symbol_date_dup_result.head().to_dict("records"),
                    }
                )
            else:
                self.logger_system.log_finding(
                    "SUCCESS", "No symbol-date duplicates found in warehouse"
                )

            # 4. Check for partial duplicates (same date, different prices)
            partial_dup_query = f"""
            SELECT date, COUNT(DISTINCT close) as distinct_closes, COUNT(*) as total_rows
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = '{symbol}'
            GROUP BY date
            HAVING COUNT(*) > 1 AND COUNT(DISTINCT close) > 1
            ORDER BY date DESC
            """

            partial_dup_result = self.raw_client.query(partial_dup_query).to_dataframe()
            duplicate_report["partial_duplicates"] = len(partial_dup_result)

            if duplicate_report["partial_duplicates"] > 0:
                self.logger_system.log_finding(
                    "WARNING",
                    f"Found {duplicate_report['partial_duplicates']} dates with multiple different prices for {symbol}",
                )
                duplicate_report["duplicate_details"].append(
                    {
                        "type": "partial_duplicates",
                        "count": duplicate_report["partial_duplicates"],
                        "sample": partial_dup_result.head().to_dict("records"),
                    }
                )
            else:
                self.logger_system.log_finding(
                    "SUCCESS",
                    f"No partial duplicates (same date, different prices) found for {symbol}",
                )

            # Summary
            total_issues = sum(
                [
                    duplicate_report["date_duplicates"],
                    duplicate_report["exact_duplicates"],
                    duplicate_report["symbol_date_duplicates"],
                    duplicate_report["partial_duplicates"],
                ]
            )

            if total_issues == 0:
                self.logger_system.log_finding(
                    "SUCCESS",
                    "Comprehensive duplicate analysis: CLEAN",
                    duplicate_report,
                )
            else:
                self.logger_system.log_finding(
                    "ERROR",
                    f"Comprehensive duplicate analysis: {total_issues} issues found",
                    duplicate_report,
                )

            return duplicate_report

        except Exception as e:
            self.logger_system.log_finding("ERROR", f"Duplicate analysis failed: {e}")
            return duplicate_report

    def check_data_quality(self, symbol: str = "AAPL") -> Dict:
        """Check for common data quality issues beyond duplicates."""
        self.logger_system.log_section(f"DATA QUALITY ANALYSIS FOR {symbol}")

        quality_report = {
            "null_values": {},
            "negative_prices": 0,
            "zero_volume": 0,
            "price_anomalies": [],
            "date_gaps": [],
            "volume_anomalies": [],
        }

        try:
            # Check for null values
            null_query = f"""
            SELECT 
                COUNTIF(close IS NULL) as null_close,
                COUNTIF(high IS NULL) as null_high,
                COUNTIF(low IS NULL) as null_low,
                COUNTIF(volume IS NULL) as null_volume,
                COUNT(*) as total_rows
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = '{symbol}'
            """

            null_result = self.raw_client.query(null_query).to_dataframe().iloc[0]

            for col in ["close", "high", "low", "volume"]:
                null_count = null_result[f"null_{col}"]
                quality_report["null_values"][col] = null_count

                if null_count > 0:
                    self.logger_system.log_finding(
                        "ERROR", f"Found {null_count} NULL values in {col}"
                    )
                else:
                    self.logger_system.log_finding(
                        "SUCCESS", f"No NULL values in {col}"
                    )

            # Check for negative prices
            neg_query = f"""
            SELECT COUNT(*) as negative_count
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = '{symbol}' AND (close < 0 OR high < 0 OR low < 0)
            """

            neg_result = self.raw_client.query(neg_query).to_dataframe().iloc[0]
            quality_report["negative_prices"] = neg_result["negative_count"]

            if quality_report["negative_prices"] > 0:
                self.logger_system.log_finding(
                    "ERROR",
                    f"Found {quality_report['negative_prices']} rows with negative prices",
                )
            else:
                self.logger_system.log_finding("SUCCESS", "No negative prices found")

            # Check for zero volume
            zero_vol_query = f"""
            SELECT COUNT(*) as zero_volume_count
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = '{symbol}' AND volume = 0
            """

            zero_vol_result = (
                self.raw_client.query(zero_vol_query).to_dataframe().iloc[0]
            )
            quality_report["zero_volume"] = zero_vol_result["zero_volume_count"]

            if quality_report["zero_volume"] > 0:
                self.logger_system.log_finding(
                    "WARNING",
                    f"Found {quality_report['zero_volume']} rows with zero volume",
                )
            else:
                self.logger_system.log_finding("SUCCESS", "No zero volume found")

            # Check for price anomalies (high < low, close outside high-low range)
            anomaly_query = f"""
            SELECT date, open, high, low, close
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = '{symbol}' 
            AND (high < low OR close > high OR close < low OR open > high OR open < low)
            ORDER BY date DESC
            """

            anomaly_result = self.raw_client.query(anomaly_query).to_dataframe()
            quality_report["price_anomalies"] = (
                anomaly_result.to_dict("records") if not anomaly_result.empty else []
            )

            if len(quality_report["price_anomalies"]) > 0:
                self.logger_system.log_finding(
                    "ERROR",
                    f"Found {len(quality_report['price_anomalies'])} price anomalies",
                )
                self.logger.info("Sample price anomalies:")
                self.logger.info(anomaly_result.head().to_string(index=False))
            else:
                self.logger_system.log_finding("SUCCESS", "No price anomalies found")

            self.logger_system.log_finding(
                "SUCCESS", "Data quality analysis completed", quality_report
            )
            return quality_report

        except Exception as e:
            self.logger_system.log_finding("ERROR", f"Data quality check failed: {e}")
            return quality_report

    def compare_with_application_expectations(self, symbol: str = "AAPL") -> Dict:
        """Compare BigQuery data with what the application expects."""
        self.logger_system.log_section(f"APPLICATION INTEGRATION ANALYSIS FOR {symbol}")

        comparison = {
            "column_mapping": {},
            "data_load_success": False,
            "app_vs_raw_differences": [],
            "price_comparison": {},
            "data_samples": {},
        }

        try:
            # Get raw BigQuery data
            raw_df = self.sample_raw_data(symbol, limit=5)
            if raw_df is None:
                return comparison

            comparison["data_samples"]["raw_bigquery"] = raw_df.head(1).to_dict(
                "records"
            )[0]

            # Try to load data through application
            try:
                start_date = date.today() - timedelta(days=365 * 2)  # 2 years ago
                app_df, source = load_bigquery_data(symbol, start_date)
                comparison["data_load_success"] = True
                comparison["data_samples"]["application"] = app_df.head(1).to_dict(
                    "records"
                )[0]
                self.logger_system.log_finding(
                    "SUCCESS", "Application data loading successful"
                )

                # Compare columns
                raw_cols = set(raw_df.columns)
                app_cols = set(app_df.columns)

                self.logger.info(f"Raw BigQuery columns: {sorted(raw_cols)}")
                self.logger.info(f"Application columns: {sorted(app_cols)}")

                missing_in_app = raw_cols - app_cols
                missing_in_raw = app_cols - raw_cols

                comparison["column_mapping"] = {
                    "raw_columns": sorted(raw_cols),
                    "app_columns": sorted(app_cols),
                    "missing_in_app": sorted(missing_in_app),
                    "missing_in_raw": sorted(missing_in_raw),
                }

                if missing_in_app:
                    self.logger_system.log_finding(
                        "ERROR", f"Columns in BigQuery but not in app: {missing_in_app}"
                    )
                    comparison["app_vs_raw_differences"].append(
                        f"Missing in app: {missing_in_app}"
                    )

                if missing_in_raw:
                    self.logger_system.log_finding(
                        "ERROR", f"Columns in app but not in BigQuery: {missing_in_raw}"
                    )
                    comparison["app_vs_raw_differences"].append(
                        f"Missing in BigQuery: {missing_in_raw}"
                    )

                if not missing_in_app and not missing_in_raw:
                    self.logger_system.log_finding(
                        "SUCCESS", "Column sets match perfectly"
                    )

                # Check data values (most recent close price)
                if "close" in raw_cols and "close" in app_cols:
                    raw_close = raw_df["close"].iloc[0]
                    app_close = app_df["close"].iloc[0]
                    price_diff = abs(raw_close - app_close)
                    price_diff_pct = (price_diff / raw_close) * 100

                    comparison["price_comparison"] = {
                        "raw_close": float(raw_close),
                        "app_close": float(app_close),
                        "difference": float(price_diff),
                        "difference_percent": float(price_diff_pct),
                    }

                    if price_diff_pct < 0.1:  # Less than 0.1% difference
                        self.logger_system.log_finding(
                            "SUCCESS",
                            f"Close price values match: {raw_close} ≈ {app_close}",
                        )
                    else:
                        self.logger_system.log_finding(
                            "ERROR",
                            f"Close price mismatch: Raw=${raw_close}, App=${app_close} ({price_diff_pct:.2f}% difference)",
                        )
                        comparison["app_vs_raw_differences"].append(
                            f"Price mismatch: {price_diff_pct:.2f}% difference"
                        )

            except Exception as e:
                self.logger_system.log_finding(
                    "ERROR", f"Application data loading failed: {e}"
                )
                comparison["data_load_success"] = False
                comparison["app_vs_raw_differences"].append(f"Load error: {str(e)}")

            self.logger_system.log_finding(
                "SUCCESS", "Application integration analysis completed", comparison
            )
            return comparison

        except Exception as e:
            self.logger_system.log_finding(
                "ERROR", f"Application integration analysis failed: {e}"
            )
            return comparison

    def get_symbol_statistics(self) -> Dict:
        """Get comprehensive statistics about all symbols in the warehouse."""
        self.logger_system.log_section("WAREHOUSE SYMBOL STATISTICS")

        try:
            stats_query = f"""
            SELECT 
                symbol,
                COUNT(*) as row_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                DATE_DIFF(MAX(date), MIN(date), DAY) as date_span_days,
                AVG(close) as avg_close,
                MIN(close) as min_close,
                MAX(close) as max_close,
                AVG(volume) as avg_volume
            FROM `{self.raw_client.project}.{self.dataset_id}.{self.table_id}`
            GROUP BY symbol
            ORDER BY row_count DESC
            """

            stats_df = self.raw_client.query(stats_query).to_dataframe()

            warehouse_stats = {
                "total_symbols": len(stats_df),
                "total_rows": int(stats_df["row_count"].sum()),
                "date_range": {
                    "earliest": str(stats_df["earliest_date"].min()),
                    "latest": str(stats_df["latest_date"].max()),
                },
                "symbol_details": stats_df.to_dict("records"),
            }

            self.logger.info(f"Found {len(stats_df)} unique symbols")
            self.logger.info(
                f"Total rows across all symbols: {stats_df['row_count'].sum():,}"
            )

            self.logger.info("\nTop 10 symbols by data volume:")
            self.logger.info(
                stats_df.head(10)[
                    ["symbol", "row_count", "earliest_date", "latest_date"]
                ].to_string(index=False)
            )

            # Check for data freshness
            recent_data = stats_df[
                stats_df["latest_date"] >= (datetime.now().date() - timedelta(days=7))
            ]
            stale_data = stats_df[
                stats_df["latest_date"] < (datetime.now().date() - timedelta(days=7))
            ]

            warehouse_stats["freshness"] = {
                "recent_symbols": len(recent_data),
                "stale_symbols": len(stale_data),
            }

            self.logger.info(
                f"\nSymbols with recent data (within 7 days): {len(recent_data)}"
            )
            self.logger.info(
                f"Symbols with stale data (older than 7 days): {len(stale_data)}"
            )

            if len(stale_data) > 0:
                self.logger.info("\nStale symbols (showing first 5):")
                self.logger.info(
                    stale_data.head()[["symbol", "latest_date", "row_count"]].to_string(
                        index=False
                    )
                )
                warehouse_stats["stale_sample"] = stale_data.head().to_dict("records")

            self.logger_system.log_finding(
                "SUCCESS", "Warehouse statistics analysis completed", warehouse_stats
            )
            return warehouse_stats

        except Exception as e:
            self.logger_system.log_finding("ERROR", f"Statistics query failed: {e}")
            return {}

    def run_full_diagnostic(self, symbol: str = "AAPL") -> Dict:
        """Run complete diagnostic suite with comprehensive logging."""
        self.logger_system.log_section("FULL BIGQUERY DIAGNOSTIC SUITE")
        self.logger.info(f"Target Symbol: {symbol}")
        self.logger.info(f"Timestamp: {datetime.now()}")
        self.logger.info(f"Log File: {self.logger_system.log_file}")

        results = {
            "diagnostic_metadata": {
                "timestamp": datetime.now().isoformat(),
                "target_symbol": symbol,
                "log_file": self.logger_system.log_file,
            }
        }

        # Run all diagnostic checks
        results["connection_health"] = self.check_connection_health()
        results["schema"] = self.examine_table_schema()
        results["sample_data"] = (
            self.sample_raw_data(symbol).to_dict("records")
            if self.sample_raw_data(symbol) is not None
            else []
        )
        results["duplicate_analysis"] = self.check_comprehensive_duplicates(symbol)
        results["data_quality"] = self.check_data_quality(symbol)
        results["app_integration"] = self.compare_with_application_expectations(symbol)
        results["warehouse_stats"] = self.get_symbol_statistics()

        # Generate comprehensive summary
        self.logger_system.log_section("DIAGNOSTIC SUMMARY")

        if results["connection_health"]:
            self.logger_system.log_finding("SUCCESS", "BigQuery connection is healthy")
        else:
            self.logger_system.log_finding("ERROR", "BigQuery connection has issues")

        if results["schema"]:
            self.logger_system.log_finding(
                "SUCCESS",
                f"Schema loaded: {results['schema']['total_columns']} columns, {results['schema']['total_rows']:,} rows",
            )

        # Duplicate summary
        dup_total = sum(
            [
                results["duplicate_analysis"]["date_duplicates"],
                results["duplicate_analysis"]["exact_duplicates"],
                results["duplicate_analysis"]["symbol_date_duplicates"],
                results["duplicate_analysis"]["partial_duplicates"],
            ]
        )

        if dup_total == 0:
            self.logger_system.log_finding(
                "SUCCESS", f"No duplicate issues found for {symbol}"
            )
        else:
            self.logger_system.log_finding(
                "ERROR", f"Found {dup_total} duplicate-related issues for {symbol}"
            )

        if results["app_integration"]["data_load_success"]:
            self.logger_system.log_finding(
                "SUCCESS", f"Application can load {symbol} data successfully"
            )

            if "price_comparison" in results["app_integration"]:
                price_diff = results["app_integration"]["price_comparison"].get(
                    "difference_percent", 0
                )
                if price_diff < 0.1:
                    self.logger_system.log_finding(
                        "SUCCESS",
                        f"Price data accuracy verified (difference: {price_diff:.3f}%)",
                    )
                else:
                    self.logger_system.log_finding(
                        "ERROR",
                        f"Price data corruption detected (difference: {price_diff:.2f}%)",
                    )
        else:
            self.logger_system.log_finding(
                "ERROR", f"Application failed to load {symbol} data"
            )

        if results["warehouse_stats"]:
            stats = results["warehouse_stats"]
            self.logger_system.log_finding(
                "SUCCESS",
                f"Warehouse contains {stats['total_symbols']} symbols with {stats['total_rows']:,} total rows",
            )

            if stats["freshness"]["stale_symbols"] > 0:
                self.logger_system.log_finding(
                    "WARNING",
                    f"{stats['freshness']['stale_symbols']} symbols have stale data",
                )

        # Save detailed results to JSON
        json_file = self.logger_system.log_file.replace(".log", "_results.json")
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Detailed results saved to: {json_file}")
        self.logger_system.log_finding(
            "SUCCESS",
            "Full diagnostic completed",
            {"log_file": self.logger_system.log_file, "results_file": json_file},
        )

        return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="BigQuery Diagnostic Tool with Comprehensive Logging"
    )
    parser.add_argument(
        "--symbol", default="AAPL", help="Symbol to diagnose (default: AAPL)"
    )
    parser.add_argument(
        "--full-audit", action="store_true", help="Run full diagnostic audit"
    )
    parser.add_argument(
        "--schema-only", action="store_true", help="Only examine schema"
    )
    parser.add_argument(
        "--stats-only", action="store_true", help="Only show symbol statistics"
    )
    parser.add_argument(
        "--quality-only", action="store_true", help="Only check data quality"
    )
    parser.add_argument(
        "--duplicates-only", action="store_true", help="Only check for duplicates"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    diagnostic = BigQueryDiagnostic(args.log_level)

    if args.full_audit:
        diagnostic.run_full_diagnostic(args.symbol)
    elif args.schema_only:
        diagnostic.examine_table_schema()
    elif args.stats_only:
        diagnostic.get_symbol_statistics()
    elif args.quality_only:
        diagnostic.check_data_quality(args.symbol)
    elif args.duplicates_only:
        diagnostic.check_comprehensive_duplicates(args.symbol)
    else:
        # Default: run focused diagnostic on specified symbol
        diagnostic.run_full_diagnostic(args.symbol)


if __name__ == "__main__":
    main()
