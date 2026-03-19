"""
NASA C-MAPSS Data Loader
Handles ingestion, preprocessing, and validation of turbofan engine degradation data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CMAPSSDataLoader:
    """Loads and processes NASA C-MAPSS dataset"""
    
    # Column names for the dataset
    SENSOR_COLUMNS = [f'sensor_{i}' for i in range(1, 22)]  # 21 sensors
    OPERATING_CONDITION_COLUMNS = ['op_cond_1', 'op_cond_2', 'op_cond_3']
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader
        
        Args:
            data_dir: Path to CMAPSSData directory
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Load train and test data for a specific dataset (FD001-004)
        
        Args:
            dataset_name: One of 'FD001', 'FD002', 'FD003', 'FD004'
            
        Returns:
            train_data, test_data, RUL values
        """
        # Load raw data
        train_file = self.data_dir / f'train_{dataset_name}.txt'
        test_file = self.data_dir / f'test_{dataset_name}.txt'
        rul_file = self.data_dir / f'RUL_{dataset_name}.txt'
        
        if not all([train_file.exists(), test_file.exists(), rul_file.exists()]):
            raise FileNotFoundError(f"Dataset files not found for {dataset_name}")
        
        # The raw files are whitespace-delimited with trailing spaces on each row.
        # Using a regex separator avoids creating shifted columns from repeated spaces.
        column_names = ['engine_id', 'time_steps'] + self.OPERATING_CONDITION_COLUMNS + self.SENSOR_COLUMNS
        
        # Load data
        train_data = pd.read_csv(train_file, sep=r'\s+', header=None, names=column_names, engine='python')
        test_data = pd.read_csv(test_file, sep=r'\s+', header=None, names=column_names, engine='python')
        
        # Load RUL values
        rul_values = np.loadtxt(rul_file, dtype=int)
        
        logger.info(f"Loaded {dataset_name}: {len(train_data)} train samples, {len(test_data)} test samples")
        
        return train_data, test_data, rul_values
    
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int = 30,
        rul_offsets: np.ndarray | None = None,
        max_rul: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding window sequences from time series data
        
        Args:
            data: DataFrame with engine data
            sequence_length: Number of timesteps in each sequence
            rul_offsets: Optional per-engine RUL offsets for truncated sequences.
                For the NASA test split, this should be the provided final RUL for
                each engine from `RUL_FD00X.txt`.
            max_rul: Optional upper bound used to clip large RUL targets.
            
        Returns:
            X (sequences), y (RUL labels), engine_ids
        """
        X, y, engine_ids_list = [], [], []
        
        engines = data['engine_id'].unique()
        if rul_offsets is not None and len(rul_offsets) != len(engines):
            raise ValueError(
                f"Expected {len(engines)} RUL offsets, got {len(rul_offsets)}."
            )
        
        for engine_idx, engine_id in enumerate(engines):
            engine_data = data[data['engine_id'] == engine_id].reset_index(drop=True)
            
            # Get RUL (Remaining Useful Life)
            total_cycles = len(engine_data)
            rul_offset = int(rul_offsets[engine_idx]) if rul_offsets is not None else 0
            
            # Create sequences
            for i in range(len(engine_data) - sequence_length + 1):
                seq = engine_data.iloc[i:i + sequence_length][self.OPERATING_CONDITION_COLUMNS + self.SENSOR_COLUMNS].values
                observed_remaining = total_cycles - i - sequence_length
                rul = max(0, observed_remaining + rul_offset)  # RUL at end of sequence
                if max_rul is not None:
                    rul = min(rul, max_rul)
                
                X.append(seq)
                y.append(rul)
                engine_ids_list.append(engine_id)
        
        return np.array(X), np.array(y), np.array(engine_ids_list)
    
    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Normalize sensor data using training statistics
        
        Args:
            X_train: Training sequences
            X_test: Test sequences
            
        Returns:
            Normalized X_train, X_test, and the scaler object
        """
        # Reshape for scaling
        n_samples, n_steps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples, n_steps, n_features)
        
        # Apply same scaler to test data
        n_test_samples = X_test.shape[0]
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_test_samples, n_steps, n_features)
        
        return X_train_scaled, X_test_scaled, scaler
    
    def process_complete_pipeline(
        self,
        dataset_name: str,
        sequence_length: int = 30,
        max_rul: int | None = None,
    ) -> Dict:
        """
        Complete data processing pipeline
        
        Args:
            dataset_name: Dataset to load
            sequence_length: Sequence length for windows
            max_rul: Optional upper bound used to clip train/test RUL targets
            
        Returns:
            Dictionary with processed data
        """
        # Load data
        train_data, test_data, rul_values = self.load_dataset(dataset_name)
        
        # Create sequences
        X_train, y_train, engine_train = self.prepare_sequences(
            train_data,
            sequence_length,
            max_rul=max_rul,
        )
        X_test, y_test, engine_test = self.prepare_sequences(
            test_data,
            sequence_length,
            rul_offsets=rul_values,
            max_rul=max_rul,
        )
        
        # Normalize
        X_train, X_test, scaler = self.normalize_data(X_train, X_test)
        
        logger.info(f"Pipeline complete: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'engine_train': engine_train,
            'engine_test': engine_test,
            'scaler': scaler,
            'dataset_name': dataset_name,
            'rul_values': rul_values
        }
