"""
Main Training Pipeline
Orchestrates complete workflow from data loading to model deployment
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
import joblib
from typing import Dict
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import CMAPSSDataLoader
from src.features import PhysicsInformedFeatures
from src.baselines import BaselineModels
from src.models import RULModels
from src.uncertainty import UncertaintyEstimation
from configs.config import TrainingConfig, DataConfig, ModelConfig, get_full_pipeline_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RULPipeline:
    """Complete RUL prediction pipeline"""
    
    def __init__(self, config: TrainingConfig = None):
        """
        Initialize pipeline
        
        Args:
            config: Training configuration
        """
        if config is None:
            config = TrainingConfig()
        self.config = config
        self.results = {}
        
        # Create model save directory
        Path(self.config.model_save_dir).mkdir(exist_ok=True)
        
        logger.info("Pipeline initialized")
    
    def run_complete_pipeline(self, dataset_name: str = 'FD001') -> Dict:
        """
        Run complete pipeline for a dataset
        
        Args:
            dataset_name: Dataset to run on (FD001-FD004)
            
        Returns:
            Dictionary with all results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING COMPLETE PIPELINE FOR {dataset_name}")
        logger.info(f"{'='*60}\n")
        
        # Set random seeds
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)
        
        # 1. Load and preprocess data
        logger.info("STEP 1: Data Loading and Preprocessing")
        logger.info("-" * 40)
        data_dict = self._load_and_preprocess(dataset_name)
        
        # 2. Train baselines
        logger.info("\n\nSTEP 2: Training Baseline Models")
        logger.info("-" * 40)
        baseline_results = self._train_baselines(data_dict)
        
        # 3. Train core LSTM model
        logger.info("\n\nSTEP 3: Training Core LSTM Model")
        logger.info("-" * 40)
        lstm_results = self._train_lstm(data_dict)
        
        # 4. Train attention-enhanced LSTM
        logger.info("\n\nSTEP 4: Training Attention-Enhanced LSTM Model")
        logger.info("-" * 40)
        attention_results = self._train_attention_lstm(data_dict)
        
        # 5. Evaluate and compare
        logger.info("\n\nSTEP 5: Model Evaluation and Comparison")
        logger.info("-" * 40)
        self._compare_models(baseline_results, lstm_results, attention_results)
        
        # 6. Uncertainty estimation
        logger.info("\n\nSTEP 6: Uncertainty Estimation")
        logger.info("-" * 40)
        self._uncertainty_analysis(attention_results, data_dict)
        
        # 7. Save best model
        logger.info("\n\nSTEP 7: Saving Models")
        logger.info("-" * 40)
        self._save_models(attention_results, data_dict, dataset_name)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE COMPLETE FOR {dataset_name}")
        logger.info(f"{'='*60}\n")
        
        return {
            'data': data_dict,
            'baselines': baseline_results,
            'lstm': lstm_results,
            'attention': attention_results
        }
    
    def _load_and_preprocess(self, dataset_name: str) -> Dict:
        """Load and preprocess data"""
        loader = CMAPSSDataLoader(self.config.data.data_dir)
        data = loader.process_complete_pipeline(
            dataset_name,
            sequence_length=self.config.data.sequence_length
        )
        
        # Add physics-informed features
        if self.config.data.use_physics_features:
            logger.info("Adding physics-informed features...")
            X_train = PhysicsInformedFeatures.aggregate_features(data['X_train'], include_physics=True)
            X_test = PhysicsInformedFeatures.aggregate_features(data['X_test'], include_physics=True)
            data['X_train'] = X_train
            data['X_test'] = X_test
            logger.info(f"Features expanded to shape: {X_train.shape}")
        
        # Split into train/val/test
        n_train_samples = len(data['X_train'])
        split_idx = int(n_train_samples * self.config.data.train_val_split)
        
        data['X_train_split'] = data['X_train'][:split_idx]
        data['y_train_split'] = data['y_train'][:split_idx]
        data['X_val'] = data['X_train'][split_idx:]
        data['y_val'] = data['y_train'][split_idx:]
        
        logger.info(f"Train set: {data['X_train_split'].shape}")
        logger.info(f"Val set: {data['X_val'].shape}")
        logger.info(f"Test set: {data['X_test'].shape}")
        
        return data
    
    def _train_baselines(self, data_dict: Dict) -> Dict:
        """Train baseline models"""
        results = BaselineModels.compare_baselines(
            data_dict['X_train_split'],
            data_dict['y_train_split'],
            data_dict['X_test'],
            data_dict['y_test']
        )
        return results
    
    def _train_lstm(self, data_dict: Dict) -> Dict:
        """Train LSTM model"""
        logger.info(f"Building LSTM model with {self.config.model.lstm_units} units...")
        
        model = RULModels.build_lstm(
            input_shape=(data_dict['X_train_split'].shape[1:]),
            lstm_units=self.config.model.lstm_units,
            dropout=self.config.model.dropout
        )
        
        logger.info("Training LSTM...")
        training_result = RULModels.train_model(
            model,
            data_dict['X_train_split'],
            data_dict['y_train_split'],
            data_dict['X_val'],
            data_dict['y_val'],
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size
        )
        
        logger.info("Evaluating LSTM...")
        test_result = RULModels.evaluate_model(model, data_dict['X_test'], data_dict['y_test'])
        logger.info(f"LSTM Test RMSE: {test_result['rmse']:.4f}, R²: {test_result['r2']:.4f}")
        
        return {'model': model, 'training': training_result, 'test': test_result}
    
    def _train_attention_lstm(self, data_dict: Dict) -> Dict:
        """Train attention-enhanced LSTM model"""
        logger.info(f"Building Attention LSTM model with {self.config.model.lstm_units} units...")
        
        model = RULModels.build_attention_lstm(
            input_shape=(data_dict['X_train_split'].shape[1:]),
            lstm_units=self.config.model.lstm_units,
            dropout=self.config.model.dropout
        )
        
        logger.info("Training Attention LSTM...")
        training_result = RULModels.train_model(
            model,
            data_dict['X_train_split'],
            data_dict['y_train_split'],
            data_dict['X_val'],
            data_dict['y_val'],
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size
        )
        
        logger.info("Evaluating Attention LSTM...")
        test_result = RULModels.evaluate_model(model, data_dict['X_test'], data_dict['y_test'])
        logger.info(f"Attention LSTM Test RMSE: {test_result['rmse']:.4f}, R²: {test_result['r2']:.4f}")
        
        return {'model': model, 'training': training_result, 'test': test_result}
    
    def _compare_models(self, baseline_results: Dict, lstm_results: Dict, attention_results: Dict):
        """Compare all models"""
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*60)
        
        # Baselines
        for name, result in baseline_results.items():
            metrics = result['metrics']
            logger.info(f"{result['name']:20s} - RMSE: {metrics['rmse']:7.4f}, MAE: {metrics['mae']:7.4f}, R²: {metrics['r2']:7.4f}")
        
        # LSTM
        lstm_metrics = lstm_results['test']
        logger.info(f"{'LSTM':20s} - RMSE: {lstm_metrics['rmse']:7.4f}, MAE: {lstm_metrics['mae']:7.4f}, R²: {lstm_metrics['r2']:7.4f}")
        
        # Attention LSTM
        attention_metrics = attention_results['test']
        logger.info(f"{'Attention LSTM':20s} - RMSE: {attention_metrics['rmse']:7.4f}, MAE: {attention_metrics['mae']:7.4f}, R²: {attention_metrics['r2']:7.4f}")
    
    def _uncertainty_analysis(self, model_results: Dict, data_dict: Dict):
        """Analyze prediction uncertainty"""
        logger.info("Analyzing prediction uncertainty...")
        
        model = model_results['model']
        test_result = model_results['test']
        
        # Regression intervals
        intervals = UncertaintyEstimation.regression_interval(
            test_result['y_pred'],
            test_result['residuals'],
            confidence=0.95
        )
        
        logger.info(f"95% Prediction Interval Width: {np.mean(intervals['margin']):.2f} cycles")
        
        # Confidence analysis
        confidence = UncertaintyEstimation.prediction_with_confidence(
            test_result['y_pred'],
            test_result['residuals']
        )
        
        logger.info(f"Mean Confidence Score: {confidence['mean_confidence']:.4f}")
        logger.info(f"High Confidence Predictions: {confidence['high_confidence_ratio']:.1%}")
        
        # Risk assessment
        risk = UncertaintyEstimation.risk_assessment(
            test_result['y_pred'],
            np.full_like(test_result['y_pred'], test_result['std_residuals']),
            critical_rul=10
        )
        
        critical_count = np.sum(risk['risk_levels'] == 'CRITICAL')
        logger.info(f"Critical Risk Engines: {critical_count}/{len(test_result['y_pred'])}")
    
    def _save_models(self, model_results: Dict, data_dict: Dict, dataset_name: str):
        """Save best model and scaler"""
        model = model_results['model']
        scaler = data_dict['scaler']
        
        model_dir = Path(self.config.model_save_dir)
        model_path = model_dir / f"attention_lstm_{dataset_name}.pt"
        scaler_path = model_dir / f"scaler_{dataset_name}.pkl"
        
        # Save model
        RULModels.save_model(model, str(model_path))
        logger.info(f"Model saved: {model_path}")
        
        # Save scaler
        joblib.dump(scaler, str(scaler_path))
        logger.info(f"Scaler saved: {scaler_path}")


def main():
    """Main entry point"""
    # Use default configuration
    config = TrainingConfig()
    
    # Create pipeline
    pipeline = RULPipeline(config)
    
    # Run for FD001 dataset
    results = pipeline.run_complete_pipeline(dataset_name='FD001')
    
    return results


if __name__ == "__main__":
    main()
