#!/usr/bin/env python3
"""
Google Cloud Function Data Pipeline Debugger

This script debugs the "Required field source cannot be null" error in the
BigQuery merge operation by testing each step of the data pipeline locally.

Usage:
    python debug_scripts/gcf_data_pipeline_debug.py
    python debug_scripts/gcf_data_pipeline_debug.py --symbol AAPL
    python debug_scripts/gcf_data_pipeline_debug.py --test-merge-only
"""

import argparse
import sys
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import json

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from google.cloud import bigquery
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Make sure you've activated the environment with activate_env.bat or activate_env.ps1"
    )
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GCFPipelineDebugger:
    """Debug the Google Cloud Function data pipeline."""

    def __init__(self):
        """Initialize the debugger with BigQuery client."""
        self.project_id = os.environ.get(
            "BIGQUERY_PROJECT_ID", "stock-forecasting-tool-2025"
        )
        self.dataset_id = "stock_data"
        self.table_id = "raw_stock_data"
        self.full_table_id = f"{self.project_id}.{self.dataset_id}.{self.table_id}"

        try:
            self.client = bigquery.Client(project=self.project_id)
            logger.info(f"Connected to BigQuery project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to connect to BigQuery: {e}")
            raise

    def create_sample_data(self, symbol: str = "AAPL") -> pd.DataFrame:
        """Create sample stock data matching the GCF format."""
        logger.info(f"Creating sample data for {symbol}")

        # Create sample data that mimics what Alpha Vantage returns
        dates = pd.date_range(start="2025-09-01", end="2025-09-10", freq="D")

        sample_data = []
        for i, date in enumerate(dates):
            sample_data.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": 150.00 + i,
                    "high": 155.00 + i,
                    "low": 148.00 + i,
                    "close": 152.00 + i,
                    "volume": 1000000 + i * 10000,
                }
            )

        df = pd.DataFrame(sample_data)

        # Convert date to datetime if it isn't already
        df["date"] = pd.to_datetime(df["date"])

        logger.info(f"Created sample data with {len(df)} records")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        logger.info(f"Sample record: {df.iloc[0].to_dict()}")

        return df

    def test_data_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Test the data preprocessing step that GCF performs."""
        logger.info("Testing data preprocessing step...")

        # This mimics the preprocessing in the GCF upsert_data method
        processed_df = df.copy()

        # Add timestamp for when data was updated
        processed_df["ingested_at"] = pd.Timestamp.utcnow()

        # Add source field - required by BigQuery schema
        processed_df["source"] = "alpha_vantage"

        # Convert date to proper format (this is where issues might occur)
        processed_df["date"] = pd.to_datetime(processed_df["date"]).dt.date

        logger.info("Data preprocessing completed")
        logger.info(f"Processed data types: {processed_df.dtypes.to_dict()}")
        logger.info(f"Processed sample record: {processed_df.iloc[0].to_dict()}")

        # Check for any null values that might cause the error
        null_counts = processed_df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values: {null_counts.to_dict()}")
        else:
            logger.info("No null values found in processed data")

        return processed_df

    def test_temp_table_creation(self, df: pd.DataFrame) -> str:
        """Test creating and loading the temporary table."""
        logger.info("Testing temporary table creation...")

        temp_table_id = (
            f"{self.full_table_id}_debug_temp_{int(datetime.now().timestamp())}"
        )
        logger.info(f"Creating temporary table: {temp_table_id}")

        try:
            # Load data to temporary table
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

            job = self.client.load_table_from_dataframe(
                df, temp_table_id, job_config=job_config
            )
            job.result()  # Wait for job to complete

            logger.info(f"Successfully created temporary table with {len(df)} records")

            # Verify the temp table data
            verify_query = f"SELECT * FROM `{temp_table_id}` LIMIT 5"
            results = list(self.client.query(verify_query).result())

            logger.info(
                f"Temporary table verification: {len(results)} records retrieved"
            )
            if results:
                logger.info(f"Sample temp table record: {dict(results[0])}")

            return temp_table_id

        except Exception as e:
            logger.error(f"Failed to create temporary table: {e}")
            raise

    def test_merge_query(self, temp_table_id: str) -> None:
        """Test the MERGE query that's causing the error."""
        logger.info("Testing MERGE query...")

        # This is the exact merge query from the GCF (updated with source field)
        merge_query = f"""
        MERGE `{self.full_table_id}` AS target
        USING `{temp_table_id}` AS source
        ON target.date = source.date AND target.symbol = source.symbol
        WHEN MATCHED THEN
            UPDATE SET
                open = source.open,
                high = source.high,
                low = source.low,
                close = source.close,
                volume = source.volume,
                ingested_at = source.ingested_at,
                source = source.source
        WHEN NOT MATCHED THEN
            INSERT (date, symbol, open, high, low, close, volume, ingested_at, source)
            VALUES (source.date, source.symbol, source.open, source.high, 
                   source.low, source.close, source.volume, source.ingested_at, source.source)
        """

        logger.info("Executing MERGE query...")
        logger.debug(f"MERGE query: {merge_query}")

        try:
            job = self.client.query(merge_query)
            result = job.result()

            logger.info("MERGE query completed successfully!")
            logger.info(f"Job statistics: {job.statistics}")

            if hasattr(result, "num_dml_affected_rows"):
                logger.info(f"Rows affected: {result.num_dml_affected_rows}")

        except Exception as e:
            logger.error(f"MERGE query failed: {e}")
            logger.error(f"Error type: {type(e)}")

            # Try to get more details about the error
            if hasattr(e, "errors"):
                logger.error(f"Detailed errors: {e.errors}")

            raise

    def verify_main_table_schema(self) -> None:
        """Verify the main table exists and has the correct schema."""
        logger.info("Verifying main table schema...")

        try:
            table = self.client.get_table(self.full_table_id)
            logger.info(f"Main table exists: {self.full_table_id}")

            # Log schema
            schema_info = []
            for field in table.schema:
                schema_info.append(
                    {
                        "name": field.name,
                        "field_type": field.field_type,
                        "mode": field.mode,
                    }
                )

            logger.info(f"Table schema: {json.dumps(schema_info, indent=2)}")

            # Get record count
            count_query = f"SELECT COUNT(*) as record_count FROM `{self.full_table_id}`"
            count_result = list(self.client.query(count_query).result())
            record_count = count_result[0].record_count if count_result else 0

            logger.info(f"Table record count: {record_count}")

        except Exception as e:
            logger.error(f"Failed to verify main table: {e}")
            raise

    def cleanup_temp_table(self, temp_table_id: str) -> None:
        """Clean up the temporary table."""
        try:
            self.client.delete_table(temp_table_id)
            logger.info(f"Cleaned up temporary table: {temp_table_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp table {temp_table_id}: {e}")

    def run_full_pipeline_test(self, symbol: str = "AAPL") -> None:
        """Run the complete pipeline test."""
        logger.info(f"Starting full pipeline test for {symbol}")

        temp_table_id = None

        try:
            # Step 1: Verify main table
            self.verify_main_table_schema()

            # Step 2: Create sample data
            df = self.create_sample_data(symbol)

            # Step 3: Test data preprocessing
            processed_df = self.test_data_preprocessing(df)

            # Step 4: Test temp table creation
            temp_table_id = self.test_temp_table_creation(processed_df)

            # Step 5: Test merge query
            self.test_merge_query(temp_table_id)

            logger.info("✅ Full pipeline test completed successfully!")

        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {e}")
            raise

        finally:
            # Cleanup
            if temp_table_id:
                self.cleanup_temp_table(temp_table_id)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Debug GCF data pipeline")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol to test with")
    parser.add_argument(
        "--test-merge-only",
        action="store_true",
        help="Only test merge operation with existing temp table",
    )

    args = parser.parse_args()

    debugger = GCFPipelineDebugger()

    if args.test_merge_only:
        logger.info("Running merge-only test...")
        # This would require an existing temp table - not implemented for safety
        logger.error("Merge-only test not implemented - run full test instead")
        return

    debugger.run_full_pipeline_test(args.symbol)


if __name__ == "__main__":
    main()
