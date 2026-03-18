"""
Uncertainty Estimation
Bayesian approaches and prediction intervals for RUL predictions
Key differentiator: Real systems care about confidence, not just point estimates
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)


class UncertaintyEstimation:
    """Methods for quantifying prediction uncertainty"""
    
    @staticmethod
    def regression_interval(y_pred: np.ndarray, residuals: np.ndarray, 
                           confidence: float = 0.95) -> Dict:
        """
        Estimate prediction intervals from residuals
        Provides upper and lower bounds on RUL predictions
        
        Args:
            y_pred: Point predictions
            residuals: Training/validation residuals
            confidence: Confidence level (0.95 = 95% CI)
            
        Returns:
            Dictionary with upper and lower bounds
        """
        # Calculate critical value (z-score for normal distribution)
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # Standard error of prediction
        std_error = np.std(residuals)
        
        # Prediction interval
        margin = z_score * std_error
        upper = y_pred + margin
        lower = np.maximum(0, y_pred - margin)  # RUL cannot be negative
        
        return {
            'point_estimate': y_pred,
            'upper_bound': upper,
            'lower_bound': lower,
            'margin': margin,
            'confidence': confidence
        }
    
    @staticmethod
    def monte_carlo_dropout(model, X_test: np.ndarray, n_iterations: int = 50) -> Dict:
        """
        Monte Carlo Dropout for uncertainty estimation
        Performs multiple forward passes with dropout enabled
        
        Args:
            model: PyTorch model
            X_test: Test data
            n_iterations: Number of forward passes
            
        Returns:
            Dictionary with mean, std, and percentiles
        """
        import torch

        device = next(model.parameters()).device
        X_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
        predictions = []

        model.train()
        for _ in range(n_iterations):
            with torch.no_grad():
                pred = model(X_tensor)
            predictions.append(pred.cpu().numpy().flatten())

        model.eval()
        
        predictions = np.array(predictions)
        
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'lower_percentile': np.percentile(predictions, 2.5, axis=0),
            'upper_percentile': np.percentile(predictions, 97.5, axis=0),
            'all_predictions': predictions
        }
    
    @staticmethod
    def quantile_regression_intervals(residuals: np.ndarray, 
                                     y_pred: np.ndarray,
                                     quantiles: Tuple = (0.05, 0.95)) -> Dict:
        """
        Quantile-based prediction intervals
        Uses actual residual distribution
        
        Args:
            residuals: Model residuals
            y_pred: Point predictions
            quantiles: Lower and upper quantiles (e.g., 0.05, 0.95)
            
        Returns:
            Dictionary with prediction intervals
        """
        lower_error = np.percentile(residuals, quantiles[0] * 100)
        upper_error = np.percentile(residuals, quantiles[1] * 100)
        
        return {
            'point_estimate': y_pred,
            'lower_bound': np.maximum(0, y_pred + lower_error),
            'upper_bound': y_pred + upper_error,
            'quantiles': quantiles
        }
    
    @staticmethod
    def prediction_with_confidence(y_pred: np.ndarray, residuals: np.ndarray,
                                  confidence_threshold: float = 0.7) -> Dict:
        """
        Classify predictions as high/medium/low confidence
        
        Args:
            y_pred: Point predictions
            residuals: Model residuals
            confidence_threshold: Threshold for confidence classification
            
        Returns:
            Dictionary with confidence scores and levels
        """
        std_residuals = np.std(residuals)
        mae_residuals = np.mean(np.abs(residuals))
        
        # Normalized margin of error
        normalized_error = mae_residuals / (np.abs(y_pred) + 1e-6)
        
        # Confidence score: inverse of normalized error
        confidence_scores = 1.0 / (1.0 + normalized_error)
        
        # Classify confidence
        confidence_levels = np.where(
            confidence_scores >= confidence_threshold,
            'HIGH',
            np.where(confidence_scores >= 0.5, 'MEDIUM', 'LOW')
        )
        
        return {
            'confidence_scores': confidence_scores,
            'confidence_levels': confidence_levels,
            'mean_confidence': np.mean(confidence_scores),
            'high_confidence_ratio': np.sum(confidence_levels == 'HIGH') / len(confidence_levels)
        }
    
    @staticmethod
    def risk_assessment(y_pred: np.ndarray, std_pred: np.ndarray,
                       critical_rul: int = 10) -> Dict:
        """
        Maintenance risk assessment
        Estimates probability of failure before predicted RUL
        
        Args:
            y_pred: Predicted RUL
            std_pred: Uncertainty (std) of predictions
            critical_rul: RUL threshold for maintenance action
            
        Returns:
            Dictionary with risk levels and maintenance recommendations
        """
        from scipy import stats
        
        # Probability of failure before critical RUL
        z_score = (critical_rul - y_pred) / (std_pred + 1e-6)
        failure_prob = stats.norm.cdf(z_score)
        
        # Risk levels
        risk_levels = np.where(
            failure_prob > 0.8,
            'CRITICAL',
            np.where(failure_prob > 0.5, 'HIGH', 
                    np.where(failure_prob > 0.2, 'MEDIUM', 'LOW'))
        )
        
        # Maintenance recommendation
        recommendations = np.where(
            risk_levels == 'CRITICAL',
            'IMMEDIATE',
            np.where(risk_levels == 'HIGH', 'URGENT',
                    np.where(risk_levels == 'MEDIUM', 'SCHEDULE', 'MONITOR'))
        )
        
        return {
            'failure_probability': failure_prob,
            'risk_levels': risk_levels,
            'recommendations': recommendations,
            'critical_rul_threshold': critical_rul
        }
    
    @staticmethod
    def calibration_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                            std_pred: np.ndarray, 
                            n_bins: int = 10) -> Dict:
        """
        Analyze if prediction intervals are well-calibrated
        
        Args:
            y_true: True values
            y_pred: Predicted means
            std_pred: Predicted standard deviations
            n_bins: Number of bins for analysis
            
        Returns:
            Calibration metrics
        """
        # 68% interval capture rate (1 std deviation)
        in_1std = np.sum(np.abs(y_true - y_pred) <= std_pred) / len(y_true)
        
        # 95% interval capture rate (2 std deviations)
        in_2std = np.sum(np.abs(y_true - y_pred) <= 2 * std_pred) / len(y_true)
        
        # MACE (Mean Absolute Calibration Error)
        expected_1std = 0.68
        expected_2std = 0.95
        mace = 0.5 * (abs(in_1std - expected_1std) + abs(in_2std - expected_2std))
        
        return {
            'capture_1std': in_1std,
            'capture_2std': in_2std,
            'expected_1std': expected_1std,
            'expected_2std': expected_2std,
            'mace': mace,
            'is_calibrated': mace < 0.05
        }
