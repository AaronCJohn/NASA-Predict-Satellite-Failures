"""
Physics-Informed Feature Engineering
Creates health indicators and degradation indicators mimicking real turbofan physics
"""

import numpy as np
import pandas as pd
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class PhysicsInformedFeatures:
    """Generate physics-based features from raw sensor data"""
    
    SENSOR_COLUMNS = [f'sensor_{i}' for i in range(1, 22)]
    
    @staticmethod
    def degradation_rate(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Calculate degradation rate: slope of sensor values over time
        Mimics how turbofan degradation accelerates
        
        Args:
            data: (n_samples, n_timesteps, n_features)
            window_size: Window for calculating slope
            
        Returns:
            (n_samples, n_timesteps, n_features) degradation rates
        """
        n_samples, n_timesteps, n_features = data.shape
        degradation = np.zeros_like(data)
        
        for i in range(n_samples):
            for t in range(window_size, n_timesteps):
                # Calculate slope over window
                degradation[i, t, :] = (data[i, t, :] - data[i, t - window_size, :]) / window_size
        
        return degradation
    
    @staticmethod
    def rolling_std(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Rolling standard deviation: captures anomalies and instability
        Higher STD = more unstable sensor = degraded engine
        
        Args:
            data: (n_samples, n_timesteps, n_features)
            window_size: Window for STD calculation
            
        Returns:
            (n_samples, n_timesteps, n_features) rolling STD
        """
        n_samples, n_timesteps, n_features = data.shape
        rolling_std = np.zeros_like(data)
        
        for i in range(n_samples):
            for t in range(window_size, n_timesteps):
                rolling_std[i, t, :] = np.std(data[i, t - window_size:t, :], axis=0)
        
        return rolling_std
    
    @staticmethod
    def cumulative_degradation(data: np.ndarray) -> np.ndarray:
        """
        Cumulative sum of degradation
        Represents total accumulated damage
        
        Args:
            data: (n_samples, n_timesteps, n_features)
            
        Returns:
            (n_samples, n_timesteps, n_features) cumulative degradation
        """
        # Normalize to 0-1 and compute cumulative sum
        degradation = PhysicsInformedFeatures.degradation_rate(data)
        
        # Make degradation positive (absolute) and cumulative
        cum_degradation = np.abs(degradation).copy()
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                cum_degradation[i, :, j] = np.cumsum(cum_degradation[i, :, j])
        
        return cum_degradation
    
    @staticmethod
    def health_indicator(data: np.ndarray) -> np.ndarray:
        """
        Combined health indicator: 1.0 = healthy, 0.0 = failed
        Based on deviation from baseline (first timesteps)
        
        Args:
            data: (n_samples, n_timesteps, n_features)
            
        Returns:
            (n_samples, n_timesteps, 1) health scores
        """
        n_samples, n_timesteps, n_features = data.shape
        baseline_window = min(5, n_timesteps // 4)
        
        health = np.zeros((n_samples, n_timesteps, 1))
        
        for i in range(n_samples):
            # Baseline is first few timesteps (healthiest state)
            baseline = np.mean(data[i, :baseline_window, :], axis=0)
            
            for t in range(n_timesteps):
                # Deviation from baseline (normalized)
                deviation = np.mean(np.abs(data[i, t, :] - baseline) / (np.abs(baseline) + 1e-6))
                # Health score: 1 - deviation (bounded to [0, 1])
                health[i, t, 0] = max(0.0, 1.0 - deviation)
        
        return health
    
    @staticmethod
    def oscillation_index(data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Capture oscillations and instability (second derivative)
        Higher oscillation = worse degradation
        
        Args:
            data: (n_samples, n_timesteps, n_features)
            window_size: Window for calculation
            
        Returns:
            (n_samples, n_timesteps, n_features) oscillation index
        """
        # Calculate first derivative
        first_deriv = np.gradient(data, axis=1)
        # Calculate second derivative (curvature/oscillation)
        oscillation = np.gradient(first_deriv, axis=1)
        
        return np.abs(oscillation)
    
    @staticmethod
    def aggregate_features(data: np.ndarray, include_physics: bool = True) -> np.ndarray:
        """
        Extract both raw and physics-informed features
        
        Args:
            data: (n_samples, n_timesteps, n_features) raw sensor data
            include_physics: Whether to include physics-informed features
            
        Returns:
            (n_samples, n_timesteps, n_total_features) combined features
        """
        if not include_physics:
            return data
        
        features = [data]  # Start with raw data
        
        # Add physics-informed features
        logger.info("Computing physics-informed features...")
        features.append(PhysicsInformedFeatures.degradation_rate(data))
        features.append(PhysicsInformedFeatures.rolling_std(data))
        features.append(PhysicsInformedFeatures.health_indicator(data))
        features.append(PhysicsInformedFeatures.oscillation_index(data))
        
        # Stack along feature dimension
        combined = np.concatenate(features, axis=2)
        logger.info(f"Feature extraction complete: {combined.shape}")
        
        return combined
