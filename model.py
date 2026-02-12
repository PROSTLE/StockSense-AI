"""
LSTM-based stock price predictor (v2.0).
Multi-feature model using Close + technical indicators.
Trains with mini-batches, LR scheduling, gradient clipping, and early stopping.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# ── Feature columns used by the model ─────────────────────────
FEATURE_COLS = ["Close", "SMA_20", "EMA_20", "RSI", "MACD", "Volume"]
LOOK_BACK = 60          # 60 trading days as input window
PREDICT_DAYS = 5


# ── LSTM Architecture ─────────────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 128,
                 num_layers: int = 3, output_size: int = 5, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

        # Deeper head: hidden → hidden/2 → output
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.head(out[:, -1, :])   # last time-step
        return out


# ── Data helpers ──────────────────────────────────────────────
def _prepare_features(df: pd.DataFrame):
    """
    Extract feature matrix and target vector from a DataFrame
    that already contains technical indicators.
    """
    # Ensure required columns exist; fall back to Close-only if not
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 2:
        available = ["Close"]

    features = df[available].values.astype(np.float32)
    close = df["Close"].values.astype(np.float32)
    return features, close, len(available)


def _prepare_data(features: np.ndarray, close_col_idx: int = 0):
    """
    Scale features, build sliding-window X/y arrays.
    Returns X, y, scaler, and the split index so the scaler is fit
    ONLY on training data (no data leakage).
    """
    n_samples = len(features)
    split = int(n_samples * 0.85)

    # Fit scaler on training portion ONLY
    scaler = MinMaxScaler()
    scaler.fit(features[:split])

    scaled = scaler.transform(features)

    X, y = [], []
    # y targets are next-5-day Close prices (column 0 after scaling)
    for i in range(LOOK_BACK, len(scaled) - PREDICT_DAYS):
        X.append(scaled[i - LOOK_BACK:i])               # shape (LOOK_BACK, n_features)
        y.append(scaled[i:i + PREDICT_DAYS, close_col_idx])  # shape (PREDICT_DAYS,)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Adjust split index for the windowed arrays
    adj_split = split - LOOK_BACK
    adj_split = max(1, min(adj_split, len(X) - 1))

    return X, y, scaler, adj_split


# ── Train + Predict pipeline ─────────────────────────────────
def train_and_predict(
    df: pd.DataFrame,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    patience: int = 10,
):
    """
    Train an LSTM on multi-feature data and return the next 5 predicted
    prices, a confidence score, and risk level.
    """
    features, close, n_features = _prepare_features(df)
    X, y, scaler, split = _prepare_data(features)

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # Torch tensors
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    # Mini-batch DataLoader
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = StockLSTM(
        input_size=n_features,
        hidden_size=128,
        num_layers=3,
        output_size=PREDICT_DAYS,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )

    # ── Training loop with early stopping ──
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_t)
            val_loss = float(criterion(val_preds, y_test_t))

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Restore best model
    if best_val_loss < float("inf"):
        model.load_state_dict(best_state)

    # ── Final test evaluation ──
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).numpy()
        test_loss = float(criterion(torch.tensor(test_preds), y_test_t))

    # Confidence & risk (based on test MSE, lower is better)
    confidence = max(0, min(100, round((1 - test_loss * 10) * 100, 1)))
    if confidence >= 70:
        risk = "Low"
    elif confidence >= 45:
        risk = "Medium"
    else:
        risk = "High"

    # ── Predict next 5 days ──
    last_window = features[-LOOK_BACK:]
    last_scaled = scaler.transform(last_window).reshape(1, LOOK_BACK, n_features)
    last_seq = torch.tensor(last_scaled, dtype=torch.float32)

    with torch.no_grad():
        raw_pred = model(last_seq).numpy().flatten()

    # Inverse-transform: raw_pred is in scaled Close space (column 0)
    # Build a dummy array with all features set to 0, then set column 0
    dummy = np.zeros((len(raw_pred), n_features), dtype=np.float32)
    dummy[:, 0] = raw_pred
    predicted_prices = scaler.inverse_transform(dummy)[:, 0].tolist()

    # Historical last 30 days for chart overlay
    historical_prices = close[-30:].tolist()

    return {
        "predicted_prices": [round(p, 2) for p in predicted_prices],
        "confidence": confidence,
        "risk": risk,
        "test_mse": round(test_loss, 6),
        "historical_last_30": [round(float(p), 2) for p in historical_prices],
    }