"""
BigQuery Optimal Parameters Table Management

This module handles the creation and management of the optimal_parameters table
in BigQuery for storing symbol-specific Prophet model parameters.

The table stores the best hyperparameters found through optimization for each symbol,
enabling faster forecasting by skipping the optimization step during real-time use.

Table Schema:
- symbol: Stock symbol (STRING, REQUIRED)
- changepoint_prior_scale: Prophet changepoint parameter (FLOAT, REQUIRED)
- seasonality_prior_scale: Prophet seasonality parameter (FLOAT, REQUIRED)
- rmse: Cross-validation RMSE performance metric (FLOAT, REQUIRED)
- mae: Cross-validation MAE performance metric (FLOAT, REQUIRED)
- smape: Cross-validation SMAPE performance metric (FLOAT, REQUIRED)
- optimization_date: When optimization was performed (DATE, REQUIRED)
- data_points_used: Number of data points in training set (INT, REQUIRED)
- cv_folds: Number of cross-validation folds used (INT, REQUIRED)
- created_at: Record creation timestamp (TIMESTAMP, REQUIRED)
- updated_at: Record last update timestamp (TIMESTAMP, REQUIRED)

Author: Shane
Created: 2025-09-21
"""

import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
from google.cloud import bigquery

# Import existing BigQuery configuration
from app_modules.bigquery_client import PROJECT_ID, DATASET_ID

logger = logging.getLogger(__name__)

# Table configuration
OPTIMAL_PARAMS_TABLE_ID = "optimal_parameters"


class OptimalParametersManager:
    """Manages the optimal_parameters table in BigQuery for symbol-specific Prophet parameters."""

    def __init__(self):
        """Initialize the manager with BigQuery client."""
        self.client = bigquery.Client(project=PROJECT_ID)
        self.dataset_id = DATASET_ID
        self.table_id = OPTIMAL_PARAMS_TABLE_ID
        self.full_table_id = f"{PROJECT_ID}.{DATASET_ID}.{OPTIMAL_PARAMS_TABLE_ID}"

    def create_table(self) -> bool:
        """
        Create the optimal_parameters table if it doesn't exist.

        Returns:
            bool: True if table was created or already exists, False on error
        """
        try:
            # Check if table already exists
            try:
                self.client.get_table(self.full_table_id)
                logger.info(f"Table {self.full_table_id} already exists")
                return True
            except Exception:
                logger.info(f"Creating table {self.full_table_id}")

            # Define table schema
            schema = [
                bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField(
                    "changepoint_prior_scale", "FLOAT", mode="REQUIRED"
                ),
                bigquery.SchemaField(
                    "seasonality_prior_scale", "FLOAT", mode="REQUIRED"
                ),
                bigquery.SchemaField("rmse", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("mae", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("smape", "FLOAT", mode="REQUIRED"),
                bigquery.SchemaField("optimization_date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("data_points_used", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("cv_folds", "INTEGER", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ]

            # Create table
            table = bigquery.Table(self.full_table_id, schema=schema)
            table = self.client.create_table(table)

            logger.info(f"Successfully created table {self.full_table_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create optimal_parameters table: {e}")
            return False

    def upsert_parameters(
        self,
        symbol: str,
        changepoint_prior_scale: float,
        seasonality_prior_scale: float,
        rmse: float,
        mae: float,
        smape: float,
        data_points_used: int,
        cv_folds: int,
    ) -> bool:
        """
        Insert or update optimal parameters for a symbol.

        Args:
            symbol: Stock symbol
            changepoint_prior_scale: Prophet changepoint parameter
            seasonality_prior_scale: Prophet seasonality parameter
            rmse: Cross-validation RMSE
            mae: Cross-validation MAE
            smape: Cross-validation SMAPE
            data_points_used: Number of training data points
            cv_folds: Number of cross-validation folds

        Returns:
            bool: True if successful, False on error
        """
        try:
            current_time = datetime.utcnow()
            optimization_date = date.today()

            # Use MERGE to upsert (insert or update)
            merge_query = f"""
            MERGE `{self.full_table_id}` AS target
            USING (
                SELECT 
                    @symbol AS symbol,
                    @changepoint_prior_scale AS changepoint_prior_scale,
                    @seasonality_prior_scale AS seasonality_prior_scale,
                    @rmse AS rmse,
                    @mae AS mae,
                    @smape AS smape,
                    @optimization_date AS optimization_date,
                    @data_points_used AS data_points_used,
                    @cv_folds AS cv_folds,
                    @current_time AS created_at,
                    @current_time AS updated_at
            ) AS source
            ON target.symbol = source.symbol
            WHEN MATCHED THEN
                UPDATE SET
                    changepoint_prior_scale = source.changepoint_prior_scale,
                    seasonality_prior_scale = source.seasonality_prior_scale,
                    rmse = source.rmse,
                    mae = source.mae,
                    smape = source.smape,
                    optimization_date = source.optimization_date,
                    data_points_used = source.data_points_used,
                    cv_folds = source.cv_folds,
                    updated_at = source.updated_at
            WHEN NOT MATCHED THEN
                INSERT (symbol, changepoint_prior_scale, seasonality_prior_scale, 
                       rmse, mae, smape, optimization_date, data_points_used, 
                       cv_folds, created_at, updated_at)
                VALUES (source.symbol, source.changepoint_prior_scale, source.seasonality_prior_scale,
                       source.rmse, source.mae, source.smape, source.optimization_date,
                       source.data_points_used, source.cv_folds, source.created_at, source.updated_at)
            """

            # Configure query parameters
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                    bigquery.ScalarQueryParameter(
                        "changepoint_prior_scale", "FLOAT", changepoint_prior_scale
                    ),
                    bigquery.ScalarQueryParameter(
                        "seasonality_prior_scale", "FLOAT", seasonality_prior_scale
                    ),
                    bigquery.ScalarQueryParameter("rmse", "FLOAT", rmse),
                    bigquery.ScalarQueryParameter("mae", "FLOAT", mae),
                    bigquery.ScalarQueryParameter("smape", "FLOAT", smape),
                    bigquery.ScalarQueryParameter(
                        "optimization_date", "DATE", optimization_date
                    ),
                    bigquery.ScalarQueryParameter(
                        "data_points_used", "INTEGER", data_points_used
                    ),
                    bigquery.ScalarQueryParameter("cv_folds", "INTEGER", cv_folds),
                    bigquery.ScalarQueryParameter(
                        "current_time", "TIMESTAMP", current_time
                    ),
                ]
            )

            # Execute merge query
            query_job = self.client.query(merge_query, job_config=job_config)
            query_job.result()  # Wait for completion

            logger.info(f"Successfully upserted optimal parameters for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert parameters for {symbol}: {e}")
            return False

    def get_parameters(self, symbol: str) -> Optional[Dict]:
        """
        Get optimal parameters for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with parameters if found, None otherwise
        """
        try:
            query = f"""
            SELECT 
                symbol,
                changepoint_prior_scale,
                seasonality_prior_scale,
                rmse,
                mae,
                smape,
                optimization_date,
                data_points_used,
                cv_folds,
                created_at,
                updated_at
            FROM `{self.full_table_id}`
            WHERE symbol = @symbol
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                ]
            )

            result = self.client.query(query, job_config=job_config).result()

            for row in result:
                return {
                    "symbol": row.symbol,
                    "changepoint_prior_scale": row.changepoint_prior_scale,
                    "seasonality_prior_scale": row.seasonality_prior_scale,
                    "rmse": row.rmse,
                    "mae": row.mae,
                    "smape": row.smape,
                    "optimization_date": row.optimization_date,
                    "data_points_used": row.data_points_used,
                    "cv_folds": row.cv_folds,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }

            logger.info(f"No optimal parameters found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Failed to get parameters for {symbol}: {e}")
            return None

    def get_all_symbols_with_parameters(self) -> List[str]:
        """
        Get list of all symbols that have optimal parameters.

        Returns:
            List of symbols with optimal parameters
        """
        try:
            query = f"""
            SELECT DISTINCT symbol
            FROM `{self.full_table_id}`
            ORDER BY symbol
            """

            result = self.client.query(query).result()
            symbols = [row.symbol for row in result]

            logger.info(f"Found optimal parameters for {len(symbols)} symbols")
            return symbols

        except Exception as e:
            logger.error(f"Failed to get symbols with parameters: {e}")
            return []

    def get_optimization_summary(self) -> pd.DataFrame:
        """
        Get summary of all optimal parameters for analysis.

        Returns:
            DataFrame with optimization summary
        """
        try:
            query = f"""
            SELECT 
                symbol,
                changepoint_prior_scale,
                seasonality_prior_scale,
                rmse,
                mae,
                smape,
                optimization_date,
                data_points_used,
                cv_folds
            FROM `{self.full_table_id}`
            ORDER BY rmse ASC
            """

            result = self.client.query(query).to_dataframe()
            logger.info(f"Retrieved optimization summary for {len(result)} symbols")
            return result

        except Exception as e:
            logger.error(f"Failed to get optimization summary: {e}")
            return pd.DataFrame()

    def delete_symbol_parameters(self, symbol: str) -> bool:
        """
        Delete optimal parameters for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            bool: True if successful, False on error
        """
        try:
            query = f"""
            DELETE FROM `{self.full_table_id}`
            WHERE symbol = @symbol
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                ]
            )

            query_job = self.client.query(query, job_config=job_config)
            result = query_job.result()

            # Check if any rows were deleted
            rows_deleted = getattr(result, "num_dml_affected_rows", 0)

            if rows_deleted > 0:
                logger.info(f"Deleted optimal parameters for {symbol}")
                return True
            else:
                logger.warning(f"No parameters found to delete for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete parameters for {symbol}: {e}")
            return False


# Convenience functions
def get_optimal_parameters_manager() -> OptimalParametersManager:
    """Get OptimalParametersManager instance."""
    return OptimalParametersManager()


def create_optimal_parameters_table() -> bool:
    """Create the optimal parameters table."""
    manager = get_optimal_parameters_manager()
    return manager.create_table()


if __name__ == "__main__":
    """Test the optimal parameters table creation."""
    print("Creating optimal parameters table...")

    manager = OptimalParametersManager()
    success = manager.create_table()

    if success:
        print("SUCCESS: Optimal parameters table ready!")

        # Test with sample data
        print("\nTesting with sample data...")
        test_success = manager.upsert_parameters(
            symbol="AAPL",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            rmse=15.25,
            mae=12.10,
            smape=8.5,
            data_points_used=500,
            cv_folds=5,
        )

        if test_success:
            print("SUCCESS: Sample data inserted!")

            # Retrieve sample data
            params = manager.get_parameters("AAPL")
            if params:
                print(f"Retrieved parameters: {params}")
            else:
                print("Failed to retrieve sample data")
        else:
            print("Failed to insert sample data")
    else:
        print("Failed to create optimal parameters table")
