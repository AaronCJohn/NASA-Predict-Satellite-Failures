"""
Baseline Models
Linear Regression, Random Forest, and Simple MLP
These establish performance baseline before moving to advanced models
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class BaselineModels:
    """Collection of baseline models for RUL prediction"""
    
    @staticmethod
    def linear_regression(X_train: np.ndarray, y_train: np.ndarray, 
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Linear Regression baseline
        
        Args:
            X_train, y_train, X_test, y_test: Training and test data
            
        Returns:
            Dictionary with model and evaluation metrics
        """
        # Flatten sequences for linear model
        n_train_samples = X_train.shape[0]
        X_train_flat = X_train.reshape(n_train_samples, -1)
        
        n_test_samples = X_test.shape[0]
        X_test_flat = X_test.reshape(n_test_samples, -1)
        
        model = LinearRegression()
        model.fit(X_train_flat, y_train)
        
        y_pred = model.predict(X_test_flat)
        
        metrics = BaselineModels._evaluate(y_test, y_pred)
        
        logger.info(f"Linear Regression - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return {
            'model': model,
            'name': 'Linear Regression',
            'y_pred': y_pred,
            'metrics': metrics
        }
    
    @staticmethod
    def random_forest(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     n_estimators: int = 100) -> Dict:
        """
        Random Forest baseline
        
        Args:
            X_train, y_train, X_test, y_test: Training and test data
            n_estimators: Number of trees
            
        Returns:
            Dictionary with model and evaluation metrics
        """
        # Flatten sequences for tree-based model
        n_train_samples = X_train.shape[0]
        X_train_flat = X_train.reshape(n_train_samples, -1)
        
        n_test_samples = X_test.shape[0]
        X_test_flat = X_test.reshape(n_test_samples, -1)
        
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
        model.fit(X_train_flat, y_train)
        
        y_pred = model.predict(X_test_flat)
        
        metrics = BaselineModels._evaluate(y_test, y_pred)
        
        logger.info(f"Random Forest - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return {
            'model': model,
            'name': 'Random Forest',
            'y_pred': y_pred,
            'metrics': metrics
        }
    
    @staticmethod
    def simple_mlp(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Simple fully-connected MLP
        
        Args:
            X_train, y_train, X_test, y_test: Training and test data
            epochs: Training epochs
            batch_size: Batch size
            
        Returns:
            Dictionary with model and evaluation metrics
        """
        # Flatten sequences for MLP
        n_train_samples = X_train.shape[0]
        X_train_flat = X_train.reshape(n_train_samples, -1)
        
        n_test_samples = X_test.shape[0]
        X_test_flat = X_test.reshape(n_test_samples, -1)
        
        input_dim = X_train_flat.shape[1]
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        history = model.fit(
            X_train_flat, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        y_pred = model.predict(X_test_flat, verbose=0).flatten()
        
        metrics = BaselineModels._evaluate(y_test, y_pred)
        
        logger.info(f"Simple MLP - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return {
            'model': model,
            'name': 'Simple MLP',
            'y_pred': y_pred,
            'metrics': metrics,
            'history': history.history
        }
    
    @staticmethod
    def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Standard evaluation metrics
        
        Args:
            y_true: Ground truth values
            y_pred: Predictions
            
        Returns:
            Dictionary with metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    @staticmethod
    def compare_baselines(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train and compare all baseline models
        
        Args:
            X_train, y_train, X_test, y_test: Training and test data
            
        Returns:
            Dictionary with all baseline results
        """
        logger.info("Training baseline models...")
        
        results = {
            'linear_regression': BaselineModels.linear_regression(X_train, y_train, X_test, y_test),
            'random_forest': BaselineModels.random_forest(X_train, y_train, X_test, y_test),
            'simple_mlp': BaselineModels.simple_mlp(X_train, y_train, X_test, y_test)
        }
        
        logger.info("\n=== BASELINE COMPARISON ===")
        for name, result in results.items():
            metrics = result['metrics']
            logger.info(f"{result['name']}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
        
        return results
