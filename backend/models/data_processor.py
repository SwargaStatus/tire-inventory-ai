import pandas as pd
import numpy as np
from typing import Any, Dict
import logging
from backend.utils.supabase_client import get_column_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def process_training_data(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Cleans and processes the raw DataFrame for ML models.
    - Handles missing values, type conversions, feature engineering, etc.
    - Returns dicts for LSTM, FFNN, and baseline models.
    - Validates columns and logs data shapes.
    """
    expected_columns = set(get_column_names())
    actual_columns = set(df.columns)
    logger.info(f"Processing training data. Data shape: {df.shape}. Columns: {list(df.columns)}")
    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns
    if missing:
        logger.error(f"Missing expected columns: {missing}")
        raise ValueError(f"Missing expected columns: {missing}")
    if extra:
        logger.warning(f"Extra columns in data: {extra}")
    # Example: fill missing values
    df = df.fillna(0)
    # Convert date columns
    if 'Snapshot_Date' in df.columns:
        df['Snapshot_Date'] = pd.to_datetime(df['Snapshot_Date'])
    # Feature engineering (add more as needed)
    # Example: create a 'demand' column as a target
    if 'OnHandQuantity' in df.columns and 'OnOrderQuantity' in df.columns:
        df['demand'] = df['OnHandQuantity'] + df['OnOrderQuantity']
    # Prepare data for LSTM (time series)
    lstm_features = [
        'OnHandQuantity', 'OnOrderQuantity', 'CommittedQuantity', 'Retail', 'Net',
        'MaximumQuantity', 'MinimumQuantity', 'Supplier_Stock', 'Quantity7D',
        'Quantity30D', 'Returns30D', 'DaysSinceLastReceipt', 'LeadTime', 'GrossMargin',
        '90DayFillRate', 'MB_current_temp', 'ON_current_temp', 'SK_current_temp',
        'MB_daily_total_snow', 'ON_daily_total_snow', 'SK_daily_total_snow',
        'MB_chance_of_snow', 'ON_chance_of_snow', 'SK_chance_of_snow'
    ]
    try:
        lstm_data = df[lstm_features].values.astype(np.float32)
    except Exception as e:
        logger.error(f"Error extracting LSTM features: {e}")
        raise ValueError(f"Error extracting LSTM features: {e}")
    # Prepare data for FFNN (static features)
    ffnn_features = [
        'IsWinterTire', 'Discontinued', 'OnSale', 'Supplier_Stock', 'GrossMargin',
        'LeadTime', 'MaximumQuantity', 'MinimumQuantity', '90DayFillRate'
    ]
    try:
        ffnn_data = df[ffnn_features].values.astype(np.float32)
    except Exception as e:
        logger.error(f"Error extracting FFNN features: {e}")
        raise ValueError(f"Error extracting FFNN features: {e}")
    # Baseline (e.g., mean demand)
    try:
        baseline_target = df['demand'].values.astype(np.float32)
    except Exception as e:
        logger.error(f"Error extracting baseline target: {e}")
        raise ValueError(f"Error extracting baseline target: {e}")
    logger.info(f"Processed data shapes - LSTM: {lstm_data.shape}, FFNN: {ffnn_data.shape}, Baseline: {baseline_target.shape}")
    return {
        'lstm_data': lstm_data,
        'ffnn_data': ffnn_data,
        'baseline_target': baseline_target,
        'raw': df
    } 