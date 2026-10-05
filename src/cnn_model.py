"""Small 1D CNN for remaining useful life. CPU by default."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import CNN_BATCH, CNN_EPOCHS, CNN_LR, CNN_PATIENCE, CNN_WINDOW, SEED
from src.hardware import configure_torch_threads


class RULCNN(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv1(x))
        hidden = torch.relu(self.conv2(hidden))
        hidden = self.pool(hidden).squeeze(-1)
        return self.fc(hidden).squeeze(-1)


def build_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    window: int = CNN_WINDOW,
    target_col: str | None = None,
    last_only: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Windows are (n, channels, time). Early cycles repeat the first row so the window is full."""
    ordered = frame.sort_values(["engine_key", "cycle"])
    groups = list(ordered.groupby("engine_key", sort=False))
    n_features = len(feature_cols)
    if last_only:
        count = len(groups)
    else:
        count = len(ordered)
    sequences = np.empty((count, n_features, window), dtype=np.float32)
    targets = np.empty(count, dtype=np.float32) if target_col else None
    indices = np.empty(count, dtype=np.int64)
    cursor = 0
    for _, group in groups:
        values = group[feature_cols].to_numpy(dtype=np.float32)
        n_rows = len(group)
        if n_rows == 0:
            continue
        # Left-pad with the first row so every cycle, including the first, has a full window.
        pad = np.repeat(values[:1], window - 1, axis=0)
        padded = np.concatenate([pad, values], axis=0)
        row_indexes = group.index.to_numpy()
        positions = [n_rows - 1] if last_only else range(n_rows)
        for position in positions:
            sequences[cursor] = padded[position : position + window].T
            indices[cursor] = row_indexes[position]
            if targets is not None and target_col is not None:
                targets[cursor] = float(group.iloc[position][target_col])
            cursor += 1
    return sequences[:cursor], None if targets is None else targets[:cursor], indices[:cursor]


def _run_epoch(
    model: RULCNN,
    sequences: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    rng: np.random.Generator,
) -> float:
    model.train(train)
    n_rows = len(sequences)
    order = rng.permutation(n_rows) if train else np.arange(n_rows)
    total = 0.0
    seen = 0
    cursor = 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        while cursor < n_rows:
            batch_ix = order[cursor : cursor + batch_size]
            cursor += batch_size
            inputs = torch.from_numpy(sequences[batch_ix]).to(device)
            labels = torch.from_numpy(targets[batch_ix]).to(device)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            pred = model(inputs)
            loss = torch.mean((pred - labels) ** 2)
            if train and optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach()) * len(batch_ix)
            seen += len(batch_ix)
    return total / max(seen, 1)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_cnn(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
    val_df: pd.DataFrame | None = None,
    window: int = CNN_WINDOW,
    epochs: int = CNN_EPOCHS,
    patience: int = CNN_PATIENCE,
    batch_size: int = CNN_BATCH,
    lr: float = CNN_LR,
    seed: int = SEED,
    target_col: str = "rul_target",
) -> tuple[RULCNN, int]:
    if (train_df["origin"] == "test_file").any():
        raise RuntimeError("test rows leaked into CNN training")
    configure_torch_threads()
    torch.manual_seed(seed)
    sequences, targets, _ = build_sequences(train_df, feature_cols, window, target_col=target_col)
    if targets is None or len(sequences) == 0:
        raise ValueError("CNN received no training sequences")
    model = RULCNN(len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    batch = max(2, min(batch_size, len(sequences)))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 1
    wait = 0
    val_sequences = None
    val_targets = None
    if val_df is not None and len(val_df) > 0:
        if (val_df["origin"] == "test_file").any():
            raise RuntimeError("test rows leaked into CNN validation")
        val_sequences, val_targets, _ = build_sequences(val_df, feature_cols, window, target_col=target_col)
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, sequences, targets, batch, optimizer, device, True, rng)
        if val_sequences is not None and val_targets is not None and len(val_sequences) > 0:
            val_loss = _run_epoch(model, val_sequences, val_targets, batch, None, device, False, rng)
        else:
            val_loss = train_loss
        print(f"  cnn epoch {epoch}/{epochs} train_mse={train_loss:.2f} val_mse={val_loss:.2f}")
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if val_df is not None and wait >= patience:
                break
    model.load_state_dict(best_state)
    model.to(device)
    return model, best_epoch


def predict_cnn(
    model: RULCNN,
    frame: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
    window: int = CNN_WINDOW,
    batch_size: int = 256,
    last_only: bool = False,
) -> np.ndarray:
    sequences, _, indices = build_sequences(frame, feature_cols, window, target_col=None, last_only=last_only)
    model.eval()
    preds = np.empty(len(sequences), dtype=np.float64)
    with torch.inference_mode():
        cursor = 0
        while cursor < len(sequences):
            batch = sequences[cursor : cursor + batch_size]
            inputs = torch.from_numpy(batch).to(device)
            preds[cursor : cursor + len(batch)] = model(inputs).detach().cpu().numpy()
            cursor += batch_size
    aligned = pd.Series(preds, index=indices)
    # A row can appear once. Reindex back to the caller's order.
    return aligned.groupby(level=0).mean().reindex(frame.index).to_numpy(dtype=np.float64)
