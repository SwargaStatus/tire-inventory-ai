import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Any, Dict
from sklearn.metrics import mean_squared_error

# LSTM Model
class LSTMModel:
    def __init__(self, input_shape):
        self.model = keras.Sequential([
            layers.LSTM(32, input_shape=input_shape, return_sequences=False),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, epochs=10, batch_size=32):
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return history

    def predict(self, X):
        return self.model.predict(X)

# FFNN Model
class FFNNModel:
    def __init__(self, input_shape):
        self.model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, epochs=10, batch_size=32):
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        return history

    def predict(self, X):
        return self.model.predict(X)

# Statistical Baseline
class BaselineModel:
    def fit(self, y):
        self.mean = np.mean(y)
    def predict(self, X):
        return np.full((X.shape[0], 1), self.mean)

def train_ensemble_models(processed: Dict, config: Dict = None) -> Dict:
    lstm_data = processed['lstm_data']
    ffnn_data = processed['ffnn_data']
    y = processed['baseline_target']
    # Reshape for LSTM: (samples, timesteps, features)
    X_lstm = lstm_data.reshape((lstm_data.shape[0], 1, lstm_data.shape[1]))
    X_ffnn = ffnn_data
    # LSTM
    lstm_model = LSTMModel(input_shape=(X_lstm.shape[1], X_lstm.shape[2]))
    lstm_hist = lstm_model.train(X_lstm, y, epochs=config.get('epochs', 10))
    lstm_preds = lstm_model.predict(X_lstm)
    # FFNN
    ffnn_model = FFNNModel(input_shape=X_ffnn.shape[1])
    ffnn_hist = ffnn_model.train(X_ffnn, y, epochs=config.get('epochs', 10))
    ffnn_preds = ffnn_model.predict(X_ffnn)
    # Baseline
    baseline_model = BaselineModel()
    baseline_model.fit(y)
    baseline_preds = baseline_model.predict(X_ffnn)
    # Ensemble (simple average)
    ensemble_preds = (lstm_preds + ffnn_preds + baseline_preds) / 3
    # Metrics
    metrics = {
        'lstm_mse': float(mean_squared_error(y, lstm_preds)),
        'ffnn_mse': float(mean_squared_error(y, ffnn_preds)),
        'baseline_mse': float(mean_squared_error(y, baseline_preds)),
        'ensemble_mse': float(mean_squared_error(y, ensemble_preds)),
    }
    # Save models if needed (not implemented)
    return {
        'metrics': metrics,
        'lstm_model': lstm_model,
        'ffnn_model': ffnn_model,
        'baseline_model': baseline_model,
        'ensemble_preds': ensemble_preds.flatten().tolist(),
    }

def generate_recommendations(processed: Dict, models: Dict = None, config: Dict = None) -> Any:
    # Use trained models to generate recommendations
    # For demo, use ensemble predictions as recommendations
    ensemble_preds = models['ensemble_preds'] if models else []
    df = processed['raw']
    df['recommended_quantity'] = ensemble_preds
    # Return in frontend-expected format
    recommendations = df[['Item', 'recommended_quantity']].to_dict(orient='records')
    return recommendations 