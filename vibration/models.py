"""Random-forest baseline and a small 1D CNN. Test windows are not passed to fit()."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vibration.config import CNN_BATCH, CNN_LR, CNN_PATIENCE, CNN_WEIGHT_DECAY, RF_MAX_DEPTH, RF_TREES, SEED


def standardize_windows(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    np.maximum(std, 1e-6, out=std)
    return (values - mean) / std


class FaultCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=4, padding=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        hidden = self.features(batch).squeeze(-1)
        return self.fc(self.dropout(hidden))


def fit_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = RF_TREES,
    seed: int = SEED,
    n_jobs: int = 4,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(features, labels)
    return model


def _loader(windows: np.ndarray, labels: np.ndarray | None, batch: int, shuffle: bool, seed: int) -> DataLoader:
    tensor = torch.from_numpy(standardize_windows(windows)).unsqueeze(1)
    if labels is None:
        dataset = TensorDataset(tensor)
    else:
        dataset = TensorDataset(tensor, torch.from_numpy(labels.astype(np.int64)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_batch = max(1, min(batch, len(dataset)))
    # Batch-norm in train mode cannot see a batch of one.
    drop_last = bool(shuffle and len(dataset) > use_batch and len(dataset) % use_batch == 1)
    return DataLoader(
        dataset,
        batch_size=use_batch,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
        num_workers=0,
    )


def _predict_logits(model: FaultCNN, windows: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    model.eval()
    loader = _loader(windows, None, batch, shuffle=False, seed=SEED)
    pieces: list[np.ndarray] = []
    with torch.inference_mode():
        for (inputs,) in loader:
            logits = model(inputs.to(device))
            pieces.append(logits.cpu().numpy())
    return np.concatenate(pieces, axis=0)


def _macro_f1_from_logits(logits: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    from vibration.metrics import macro_f1

    predicted = logits.argmax(axis=1)
    names = [str(index) for index in range(n_classes)]
    truth = [str(int(label)) for label in labels]
    guess = [str(int(label)) for label in predicted]
    return macro_f1(truth, guess, names)


def train_cnn(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int,
    epochs: int,
    device: torch.device,
    seed: int = SEED,
    batch: int = CNN_BATCH,
    patience: int = CNN_PATIENCE,
) -> tuple[FaultCNN, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FaultCNN(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    loader = _loader(train_windows, train_labels, batch, shuffle=True, seed=seed)
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    wait = 0
    ran = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        ran = epoch + 1
        val_logits = _predict_logits(model, val_windows, device, batch)
        val_f1 = _macro_f1_from_logits(val_logits, val_labels, n_classes)
        print(f"cnn epoch {ran} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"epochs_ran": ran, "val_macro_f1": best_f1}


def predict_cnn(model: FaultCNN, windows: np.ndarray, device: torch.device, batch: int = CNN_BATCH) -> np.ndarray:
    logits = _predict_logits(model, windows, device, batch)
    return logits.argmax(axis=1).astype(np.int64)
