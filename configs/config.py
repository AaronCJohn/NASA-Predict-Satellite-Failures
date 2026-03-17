"""
Configuration for RUL prediction system
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    """Data loading configuration"""
    data_dir: str = "CMAPSSData"
    datasets: List[str] = None
    sequence_length: int = 30
    train_val_split: float = 0.8
    use_physics_features: bool = True
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ['FD001', 'FD002', 'FD003', 'FD004']


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str = "attention_lstm"  # lstm, attention_lstm, cnn_lstm
    lstm_units: int = 64
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10


@dataclass
class TrainingConfig:
    """Training configuration"""
    data: DataConfig = None
    model: ModelConfig = None
    random_seed: int = 42
    use_gpu: bool = True
    save_model: bool = True
    model_save_dir: str = "models"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()


@dataclass
class UncertaintyConfig:
    """Uncertainty estimation configuration"""
    enable_uncertainty: bool = True
    method: str = "mc_dropout"  # mc_dropout, regression_interval, quantile
    confidence: float = 0.95
    mc_iterations: int = 50
    critical_rul_threshold: int = 10


@dataclass
class APIConfig:
    """API deployment configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "models/best_model.h5"
    scaler_path: str = "models/scaler.pkl"
    use_mc_dropout: bool = True
    mc_iterations: int = 50


def get_default_config() -> TrainingConfig:
    """Get default configuration"""
    return TrainingConfig()


def get_full_pipeline_config() -> dict:
    """Configuration for complete pipeline"""
    return {
        'data': {
            'data_dir': 'CMAPSSData',
            'datasets': ['FD001'],  # Start with one dataset
            'sequence_length': 30,
            'use_physics_features': True
        },
        'baselines': {
            'train_linear_regression': True,
            'train_random_forest': True,
            'train_simple_mlp': True
        },
        'models': {
            'train_lstm': True,
            'train_attention_lstm': True,
            'train_cnn_lstm': False  # Optional
        },
        'uncertainty': {
            'enable_uncertainty': True,
            'use_mc_dropout': True,
            'n_iterations': 50
        },
        'evaluation': {
            'plot_results': True,
            'save_plots': True,
            'generate_report': True
        }
    }
