"""
LSTM and Attention-Enhanced Models
Core and advanced deep learning architectures for RUL prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """Custom Attention mechanism for LSTM outputs"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, time_steps, hidden_dim)
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=-1)
        
        # Reshape for multiplication
        ait = tf.expand_dims(ait, -1)
        weighted_input = x * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output, ait


class RULModels:
    """Collection of LSTM-based models for RUL prediction"""
    
    @staticmethod
    def build_lstm(input_shape: Tuple, lstm_units: int = 64, dropout: float = 0.2) -> keras.Model:
        """
        Build standard LSTM model
        
        Args:
            input_shape: (sequence_length, n_features)
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.LSTM(lstm_units, activation='relu', input_shape=input_shape, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(lstm_units // 2, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    @staticmethod
    def build_attention_lstm(input_shape: Tuple, lstm_units: int = 64, dropout: float = 0.2) -> keras.Model:
        """
        Build LSTM with attention mechanism
        Learns which timesteps are most important for RUL prediction
        
        Args:
            input_shape: (sequence_length, n_features)
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # First LSTM layer with return_sequences=True for attention
        x = layers.LSTM(lstm_units, activation='relu', return_sequences=True)(inputs)
        x = layers.Dropout(dropout)(x)
        
        # Second LSTM layer
        x = layers.LSTM(lstm_units // 2, activation='relu', return_sequences=True)(x)
        x = layers.Dropout(dropout)(x)
        
        # Attention layer
        attention_out, attention_weights = AttentionLayer()(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(attention_out)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape: Tuple, dropout: float = 0.2) -> keras.Model:
        """
        Build CNN + LSTM hybrid model
        CNN extracts spatial patterns, LSTM captures temporal dynamics
        
        Args:
            input_shape: (sequence_length, n_features)
            dropout: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # Reshape for Conv1D: add channel dimension
        x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
        
        # CNN layers
        x = layers.Conv2D(32, (3, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 1), activation='relu', padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Reshape((-1, 64))(x)  # Reshape for LSTM
        
        # LSTM layers
        x = layers.LSTM(64, activation='relu', return_sequences=True)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(32, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        
        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    @staticmethod
    def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 100, batch_size: int = 32,
                   early_stopping: bool = True) -> Dict:
        """
        Train a model with optional early stopping
        
        Args:
            model: Keras model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Maximum training epochs
            batch_size: Batch size
            early_stopping: Enable early stopping
            
        Returns:
            Dictionary with model, history, and performance
        """
        callbacks = []
        if early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ))
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
        
        logger.info(f"Training complete - Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        return {
            'model': model,
            'history': history.history,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae
        }
    
    @staticmethod
    def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            model: Trained Keras model
            X_test, y_test: Test data
            
        Returns:
            Dictionary with metrics and predictions
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate residuals for uncertainty
        residuals = y_test - y_pred
        
        return {
            'y_pred': y_pred,
            'y_test': y_test,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'residuals': residuals,
            'std_residuals': np.std(residuals)
        }
