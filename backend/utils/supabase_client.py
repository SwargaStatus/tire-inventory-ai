import os
import logging
from supabase import create_client, Client
import pandas as pd

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

_client = None

def get_supabase_client() -> Client:
    global _client
    if _client is None:
        logger.info("Attempting to create Supabase client...")
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.error("Supabase credentials not set in environment variables.")
            raise RuntimeError("Supabase credentials not set in environment variables.")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client created successfully.")
    return _client

def fetch_training_data(days_to_fetch: int = 90, lookback_days: int = 30) -> pd.DataFrame:
    """
    Fetches training data from Supabase 'trainingdata' table for the given window.
    Handles errors and logs details.
    """
    try:
        client = get_supabase_client()
        logger.info(f"Fetching data from 'trainingdata' table...")
        response = client.table('trainingdata').select('*').execute()
        if not response.data:
            logger.error('No data returned from Supabase')
            raise ValueError('No data returned from Supabase')
        df = pd.DataFrame(response.data)
        logger.info(f"Fetched {len(df)} rows from Supabase.")
        return df
    except Exception as e:
        logger.exception("Error fetching training data from Supabase")
        raise

def get_column_names() -> list:
    """
    Returns the expected column names for the trainingdata table.
    """
    return [
        "Snapshot_Date", "Item", "OnHandQuantity", "OnOrderQuantity",
        "CommittedQuantity", "Retail", "Net", "MaximumQuantity", "MinimumQuantity",
        "IsWinterTire", "Discontinued", "OnSale", "Supplier", "Description",
        "Supplier_Stock", "Quantity7D", "Quantity30D", "Returns30D", "DaysSinceLastReceipt",
        "LeadTime", "GrossMargin", "90DayFillRate",
        "MB_current_temp", "ON_current_temp", "SK_current_temp",
        "MB_daily_total_snow", "ON_daily_total_snow", "SK_daily_total_snow",
        "MB_chance_of_snow", "ON_chance_of_snow", "SK_chance_of_snow"
    ]

def inspect_table_schema() -> list:
    """
    Returns the actual column names from the Supabase 'trainingdata' table.
    Handles and logs errors.
    """
    try:
        client = get_supabase_client()
        logger.info("Inspecting table schema for 'trainingdata'...")
        response = client.table('trainingdata').select('*').limit(1).execute()
        if not response.data:
            logger.warning("No data found in 'trainingdata' table to inspect schema.")
            return []
        df = pd.DataFrame(response.data)
        logger.info(f"Actual columns in 'trainingdata': {list(df.columns)}")
        return list(df.columns)
    except Exception as e:
        logger.exception("Error inspecting table schema")
        raise 