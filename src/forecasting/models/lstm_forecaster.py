"""PyTorch LSTM Time-Series Forecaster (Optional Deep Learning)."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logging_config import setup_logger

logger = setup_logger("lstm_forecaster")


# ---------------------------------------------------------------------------
# Module-level PyTorch module — must be at module scope to be picklable
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    class _LSTMNet(nn.Module):
        """Single-layer LSTM network with dropout."""

        def __init__(self, hidden_dim: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

except ImportError:
    _LSTMNet = None  # type: ignore[assignment,misc]


class LSTMForecaster:
    """Simple 1-layer PyTorch LSTM for time-series forecasting."""

    def __init__(
        self,
        sequence_length: int = 14,
        hidden_dim: int = 32,
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 0.005,
        name: str = "LSTM",
    ):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.name = name

        self.model_ = None
        self.min_val_ = 0.0
        self.max_val_ = 1.0
        self.fitted_ = False
        self.is_available_ = self._check_torch()

    def _check_torch(self) -> bool:
        try:
            import torch
            return True
        except ImportError:
            logger.info("PyTorch unavailable — LSTM skipped.")
            return False

    def fit(self, y: pd.Series) -> "LSTMForecaster":
        if not self.is_available_:
            return self

        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset

            y_vals = y.dropna().values.astype(np.float32)
            if len(y_vals) < self.sequence_length + 5:
                logger.warning("Insufficient history for LSTM fit.")
                return self

            # Min-Max Scale (fitted ONLY on train)
            self.min_val_ = float(np.min(y_vals))
            self.max_val_ = float(np.max(y_vals))
            denom = self.max_val_ - self.min_val_ if (self.max_val_ - self.min_val_) > 1e-5 else 1.0
            y_norm = (y_vals - self.min_val_) / denom

            # Create sequences
            X_seq, y_target = [], []
            for i in range(len(y_norm) - self.sequence_length):
                X_seq.append(y_norm[i : i + self.sequence_length])
                y_target.append(y_norm[i + self.sequence_length])

            X_t = torch.tensor(np.array(X_seq), dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
            y_t = torch.tensor(np.array(y_target), dtype=torch.float32).unsqueeze(-1)

            dataset = TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

            # Use module-level _LSTMNet (picklable)
            import torch.nn as nn
            net = _LSTMNet(self.hidden_dim)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
            criterion = nn.MSELoss()

            net.train()
            for epoch in range(self.epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = net(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()

            self.model_ = net
            self.fitted_ = True
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
            self.fitted_ = False
        return self

    def predict_recursive(self, y_history: pd.Series, horizon: int) -> np.ndarray:
        if not self.is_available_ or not self.fitted_ or self.model_ is None:
            raise RuntimeError("LSTM is not fitted or PyTorch is unavailable.")

        import torch
        self.model_.eval()

        y_vals = list(y_history.dropna().values.astype(np.float32))
        denom = self.max_val_ - self.min_val_ if (self.max_val_ - self.min_val_) > 1e-5 else 1.0
        y_norm = list((np.array(y_vals) - self.min_val_) / denom)

        preds_norm = []
        with torch.no_grad():
            for _ in range(horizon):
                seq = y_norm[-self.sequence_length:]
                x_t = torch.tensor(seq, dtype=torch.float32).view(1, self.sequence_length, 1)
                p_norm = float(self.model_(x_t).item())
                preds_norm.append(p_norm)
                y_norm.append(p_norm)

        # Unscale
        preds_unscaled = np.array(preds_norm) * denom + self.min_val_
        return np.clip(preds_unscaled, 0.0, None)
