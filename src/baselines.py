"""
Baseline Models
Linear Regression, Random Forest, and a simple PyTorch MLP baseline.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def _torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SimpleMLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class BaselineModels:
    """Collection of baseline models for RUL prediction."""

    @staticmethod
    def linear_regression(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        n_train_samples = X_train.shape[0]
        X_train_flat = X_train.reshape(n_train_samples, -1)

        n_test_samples = X_test.shape[0]
        X_test_flat = X_test.reshape(n_test_samples, -1)

        model = LinearRegression()
        model.fit(X_train_flat, y_train)

        y_pred = model.predict(X_test_flat)
        metrics = BaselineModels._evaluate(y_test, y_pred)
        logger.info(
            "Linear Regression - RMSE: %.4f, MAE: %.4f",
            metrics["rmse"],
            metrics["mae"],
        )

        return {
            "model": model,
            "name": "Linear Regression",
            "y_pred": y_pred,
            "metrics": metrics,
        }

    @staticmethod
    def random_forest(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_estimators: int = 100,
    ) -> Dict:
        n_train_samples = X_train.shape[0]
        X_train_flat = X_train.reshape(n_train_samples, -1)

        n_test_samples = X_test.shape[0]
        X_test_flat = X_test.reshape(n_test_samples, -1)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train_flat, y_train)

        y_pred = model.predict(X_test_flat)
        metrics = BaselineModels._evaluate(y_test, y_pred)
        logger.info(
            "Random Forest - RMSE: %.4f, MAE: %.4f",
            metrics["rmse"],
            metrics["mae"],
        )

        return {
            "model": model,
            "name": "Random Forest",
            "y_pred": y_pred,
            "metrics": metrics,
        }

    @staticmethod
    def simple_mlp(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 15,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        patience: int = 3,
    ) -> Dict:
        n_train_samples = X_train.shape[0]
        X_train_flat = X_train.reshape(n_train_samples, -1).astype(np.float32)

        n_test_samples = X_test.shape[0]
        X_test_flat = X_test.reshape(n_test_samples, -1).astype(np.float32)

        y_train_np = y_train.astype(np.float32)
        y_test_np = y_test.astype(np.float32)

        val_size = max(1, int(0.1 * len(X_train_flat)))
        train_size = len(X_train_flat) - val_size
        if train_size <= 0:
            raise ValueError("Not enough training samples to create a validation split.")

        X_subtrain, X_val = X_train_flat[:train_size], X_train_flat[train_size:]
        y_subtrain, y_val = y_train_np[:train_size], y_train_np[train_size:]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_subtrain), torch.from_numpy(y_subtrain)),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=batch_size,
            shuffle=False,
        )

        device = _torch_device()
        model = SimpleMLPRegressor(X_train_flat.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        logger.info(
            "Simple MLP training on %s with flattened input shape %s",
            device,
            X_train_flat.shape,
        )

        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            model.train()
            train_losses = []

            for features, targets in train_loader:
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                predictions = model(features)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    loss = criterion(model(features), targets)
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                "Simple MLP - Epoch %d/%d - loss: %.4f - val_loss: %.4f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info("Simple MLP early stopping triggered at epoch %d", epoch + 1)
                    break

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test_flat).to(device)
            y_pred = model(X_test_tensor).cpu().numpy()

        metrics = BaselineModels._evaluate(y_test_np, y_pred)
        logger.info(
            "Simple MLP - RMSE: %.4f, MAE: %.4f",
            metrics["rmse"],
            metrics["mae"],
        )

        return {
            "model": model,
            "name": "Simple MLP",
            "y_pred": y_pred,
            "metrics": metrics,
            "history": history,
            "device": str(device),
        }

    @staticmethod
    def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    @staticmethod
    def compare_baselines(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        logger.info("Training baseline models...")

        results = {
            "linear_regression": BaselineModels.linear_regression(X_train, y_train, X_test, y_test),
            "random_forest": BaselineModels.random_forest(X_train, y_train, X_test, y_test),
            "simple_mlp": BaselineModels.simple_mlp(X_train, y_train, X_test, y_test),
        }

        logger.info("\n=== BASELINE COMPARISON ===")
        for result in results.values():
            metrics = result["metrics"]
            logger.info(
                "%s: RMSE=%.4f, MAE=%.4f, R²=%.4f",
                result["name"],
                metrics["rmse"],
                metrics["mae"],
                metrics["r2"],
            )

        return results
