import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Any, Dict, Optional
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# LSTM Model
class LSTMModel:
    def __init__(self, input_shape):
        logger.info(f"Initializing LSTMModel with input_shape={input_shape}")
        self.model = keras.Sequential([
            layers.LSTM(32, input_shape=input_shape, return_sequences=False),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, epochs=10, batch_size=32):
        logger.info(f"Training LSTMModel: X.shape={X.shape}, y.shape={y.shape}, epochs={epochs}")
        try:
            history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            logger.info("LSTMModel training complete.")
            return history
        except Exception as e:
            logger.error(f"Error training LSTMModel: {e}")
            raise

    def predict(self, X):
        logger.info(f"Predicting with LSTMModel: X.shape={X.shape}")
        try:
            preds = self.model.predict(X)
            logger.info(f"LSTMModel prediction complete. preds.shape={preds.shape}")
            return preds
        except Exception as e:
            logger.error(f"Error predicting with LSTMModel: {e}")
            raise

# FFNN Model
class FFNNModel:
    def __init__(self, input_shape):
        logger.info(f"Initializing FFNNModel with input_shape={input_shape}")
        self.model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, epochs=10, batch_size=32):
        logger.info(f"Training FFNNModel: X.shape={X.shape}, y.shape={y.shape}, epochs={epochs}")
        try:
            history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
            logger.info("FFNNModel training complete.")
            return history
        except Exception as e:
            logger.error(f"Error training FFNNModel: {e}")
            raise

    def predict(self, X):
        logger.info(f"Predicting with FFNNModel: X.shape={X.shape}")
        try:
            preds = self.model.predict(X)
            logger.info(f"FFNNModel prediction complete. preds.shape={preds.shape}")
            return preds
        except Exception as e:
            logger.error(f"Error predicting with FFNNModel: {e}")
            raise

# Statistical Baseline
class BaselineModel:
    def fit(self, y):
        logger.info(f"Fitting BaselineModel with y.shape={y.shape}")
        self.mean = np.mean(y)
    def predict(self, X):
        logger.info(f"Predicting with BaselineModel: X.shape={X.shape}")
        return np.full((X.shape[0], 1), self.mean)

def train_ensemble_models(processed: Dict, config: Optional[Dict] = None) -> Dict:
    try:
        lstm_data = processed['lstm_data']
        ffnn_data = processed['ffnn_data']
        y = processed['baseline_target']
        logger.info(f"train_ensemble_models: lstm_data.shape={lstm_data.shape}, ffnn_data.shape={ffnn_data.shape}, y.shape={y.shape}")
        # Reshape for LSTM: (samples, timesteps, features)
        X_lstm = lstm_data.reshape((lstm_data.shape[0], 1, lstm_data.shape[1]))
        X_ffnn = ffnn_data
        # LSTM
        lstm_model = LSTMModel(input_shape=(X_lstm.shape[1], X_lstm.shape[2]))
        lstm_hist = lstm_model.train(X_lstm, y, epochs=config.get('epochs', 10) if config else 10)
        lstm_preds = lstm_model.predict(X_lstm)
        # FFNN
        ffnn_model = FFNNModel(input_shape=X_ffnn.shape[1])
        ffnn_hist = ffnn_model.train(X_ffnn, y, epochs=config.get('epochs', 10) if config else 10)
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
        logger.info(f"Model training metrics: {metrics}")
        return {
            'metrics': metrics,
            'lstm_model': lstm_model,
            'ffnn_model': ffnn_model,
            'baseline_model': baseline_model,
            'ensemble_preds': ensemble_preds.flatten().tolist(),
        }
    except Exception as e:
        logger.error(f"Error in train_ensemble_models: {e}")
        raise

def generate_recommendations(processed: Dict, models: Optional[Dict] = None, config: Optional[Dict] = None) -> Any:
    try:
        ensemble_preds = models['ensemble_preds'] if models else []
        df = processed['raw']
        logger.info(f"Generating recommendations for {len(df)} rows.")
        df['recommended_quantity'] = ensemble_preds
        recommendations = df[['Item', 'recommended_quantity']].to_dict(orient='records')
        logger.info(f"Generated {len(recommendations)} recommendations.")
        return recommendations
    except Exception as e:
        logger.error(f"Error in generate_recommendations: {e}")
        raise 