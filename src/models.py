"""
PyTorch sequence models for RUL prediction.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))


class AttentionLayer(nn.Module):
    """Attention over LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.u = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.tanh(self.W(x))
        weights = torch.softmax(self.u(scores), dim=1)
        context = torch.sum(x * weights, dim=1)
        return context, weights


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, lstm_units: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm_units = lstm_units
        self.dropout_rate = dropout
        self.lstm1 = nn.LSTM(input_size, lstm_units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_units // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.model_name = "lstm"
        self.input_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])
        return self.head(x).squeeze(-1)


class AttentionLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, lstm_units: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm_units = lstm_units
        self.dropout_rate = dropout
        self.lstm1 = nn.LSTM(input_size, lstm_units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.attention = AttentionLayer(lstm_units // 2)
        self.head = nn.Sequential(
            nn.Linear(lstm_units // 2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

        self.model_name = "attention_lstm"
        self.input_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        context, _ = self.attention(x)
        return self.head(context).squeeze(-1)


class CNNLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.2):
        super().__init__()
        self.dropout_rate = dropout
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm1 = nn.LSTM(64, 64, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.model_name = "cnn_lstm"
        self.input_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])
        return self.head(x).squeeze(-1)


class RULModels:
    """Collection of PyTorch models for RUL prediction."""

    @staticmethod
    def build_lstm(input_shape: Tuple, lstm_units: int = 64, dropout: float = 0.2) -> nn.Module:
        model = LSTMRegressor(input_size=input_shape[1], lstm_units=lstm_units, dropout=dropout)
        model.input_shape = tuple(input_shape)
        return model

    @staticmethod
    def build_attention_lstm(input_shape: Tuple, lstm_units: int = 64, dropout: float = 0.2) -> nn.Module:
        model = AttentionLSTMRegressor(input_size=input_shape[1], lstm_units=lstm_units, dropout=dropout)
        model.input_shape = tuple(input_shape)
        return model

    @staticmethod
    def build_cnn_lstm(input_shape: Tuple, dropout: float = 0.2) -> nn.Module:
        model = CNNLSTMRegressor(input_size=input_shape[1], dropout=dropout)
        model.input_shape = tuple(input_shape)
        return model

    @staticmethod
    def train_model(
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
    ) -> Dict:
        device = get_torch_device()
        model = model.to(device)

        train_loader = DataLoader(
            TensorDataset(_to_float_tensor(X_train), _to_float_tensor(y_train)),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(_to_float_tensor(X_val), _to_float_tensor(y_val)),
            batch_size=batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        patience = 10 if early_stopping else epochs

        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}

        for epoch in range(epochs):
            model.train()
            train_losses = []
            train_maes = []

            for features, targets in train_loader:
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                predictions = model(features)
                loss = criterion(predictions, targets)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

                train_losses.append(loss.item())
                train_maes.append(torch.mean(torch.abs(predictions - targets)).item())

            model.eval()
            val_losses = []
            val_maes = []
            with torch.no_grad():
                for features, targets in val_loader:
                    features = features.to(device)
                    targets = targets.to(device)
                    predictions = model(features)
                    val_losses.append(criterion(predictions, targets).item())
                    val_maes.append(torch.mean(torch.abs(predictions - targets)).item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            train_mae = float(np.mean(train_maes))
            val_mae = float(np.mean(val_maes))

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["mae"].append(train_mae)
            history["val_mae"].append(val_mae)

            logger.info(
                "Epoch %d/%d - loss: %.4f - mae: %.4f - val_loss: %.4f - val_mae: %.4f",
                epoch + 1,
                epochs,
                train_loss,
                train_mae,
                val_loss,
                val_mae,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        model.load_state_dict(best_state)
        train_metrics = RULModels._evaluate_loader(model, train_loader, device)
        val_metrics = RULModels._evaluate_loader(model, val_loader, device)

        logger.info(
            "Training complete - Val Loss: %.4f, Val MAE: %.4f",
            val_metrics["loss"],
            val_metrics["mae"],
        )

        return {
            "model": model,
            "history": history,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "device": str(device),
        }

    @staticmethod
    def _evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
        criterion = nn.MSELoss()
        losses = []
        maes = []

        model.eval()
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(device)
                targets = targets.to(device)
                predictions = model(features)
                losses.append(criterion(predictions, targets).item())
                maes.append(torch.mean(torch.abs(predictions - targets)).item())

        return {"loss": float(np.mean(losses)), "mae": float(np.mean(maes))}

    @staticmethod
    def predict(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        device = next(model.parameters()).device
        loader = DataLoader(_to_float_tensor(X), batch_size=batch_size, shuffle=False)

        predictions = []
        model.eval()
        with torch.no_grad():
            for features in loader:
                if isinstance(features, (list, tuple)):
                    features = features[0]
                batch_pred = model(features.to(device)).cpu().numpy()
                predictions.append(batch_pred)

        return np.concatenate(predictions, axis=0)

    @staticmethod
    def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_pred = RULModels.predict(model, X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        residuals = y_test - y_pred

        return {
            "y_pred": y_pred,
            "y_test": y_test,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "residuals": residuals,
            "std_residuals": np.std(residuals),
        }

    @staticmethod
    def save_model(model: nn.Module, model_path: str) -> None:
        checkpoint = {
            "model_type": getattr(model, "model_name", model.__class__.__name__.lower()),
            "input_shape": getattr(model, "input_shape", None),
            "model_kwargs": {
                "lstm_units": getattr(model, "lstm_units", None),
                "dropout": getattr(model, "dropout_rate", None),
            },
            "state_dict": model.state_dict(),
        }
        torch.save(checkpoint, model_path)

    @staticmethod
    def load_model(model_path: str, map_location: Optional[str] = None) -> nn.Module:
        checkpoint = torch.load(model_path, map_location=map_location or get_torch_device())
        model_type = checkpoint["model_type"]
        input_shape = tuple(checkpoint["input_shape"])
        model_kwargs = {
            key: value
            for key, value in checkpoint.get("model_kwargs", {}).items()
            if value is not None
        }
        if model_type == "lstm":
            model = RULModels.build_lstm(input_shape, **model_kwargs)
        elif model_type == "attention_lstm":
            model = RULModels.build_attention_lstm(input_shape, **model_kwargs)
        elif model_type == "cnn_lstm":
            model = RULModels.build_cnn_lstm(input_shape, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model type in checkpoint: {model_type}")

        model.load_state_dict(checkpoint["state_dict"])
        return model
