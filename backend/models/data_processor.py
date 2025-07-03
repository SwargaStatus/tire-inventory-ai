import pandas as pd
import numpy as np
from typing import Any, Dict

def process_training_data(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Cleans and processes the raw DataFrame for ML models.
    - Handles missing values, type conversions, feature engineering, etc.
    - Returns dicts for LSTM, FFNN, and baseline models.
    """
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
    lstm_data = df[lstm_features].values.astype(np.float32)
    # Prepare data for FFNN (static features)
    ffnn_features = [
        'IsWinterTire', 'Discontinued', 'OnSale', 'Supplier_Stock', 'GrossMargin',
        'LeadTime', 'MaximumQuantity', 'MinimumQuantity', '90DayFillRate'
    ]
    ffnn_data = df[ffnn_features].values.astype(np.float32)
    # Baseline (e.g., mean demand)
    baseline_target = df['demand'].values.astype(np.float32)
    return {
        'lstm_data': lstm_data,
        'ffnn_data': ffnn_data,
        'baseline_target': baseline_target,
        'raw': df
    } 