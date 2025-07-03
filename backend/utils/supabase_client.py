import os
from supabase import create_client, Client
import pandas as pd

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

_client = None

def get_supabase_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("Supabase credentials not set in environment variables.")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client

def fetch_training_data(days_to_fetch: int = 90, lookback_days: int = 30) -> pd.DataFrame:
    """
    Fetches training data from Supabase 'trainingdata' table for the given window.
    """
    client = get_supabase_client()
    # Example: fetch all rows, you can add filters as needed
    response = client.table('trainingdata').select('*').execute()
    if not response.data:
        raise ValueError('No data returned from Supabase')
    df = pd.DataFrame(response.data)
    # Optionally filter by date or lookback here
    return df

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