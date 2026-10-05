"""Control charts on healthy-period rolling means, plus an isolation forest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import IFOREST_TREES, N_THREADS, SEED


@dataclass
class AnomalyModel:
    forest: IsolationForest
    mu: np.ndarray
    sigma: np.ndarray
    columns: list[str]
    score_thresholds: dict[str, float]


def _standardize(values: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    scaled = (values - mu) / sigma
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def fit_anomaly(
    train_df: pd.DataFrame,
    columns: list[str],
    seed: int = SEED,
    n_estimators: int = IFOREST_TREES,
    n_jobs: int = N_THREADS,
    healthy_rul: float = 80.0,
) -> AnomalyModel:
    """Fit on train engines only, using cycles that still have plenty of life left."""
    if (train_df["origin"] == "test_file").any():
        raise RuntimeError("test rows leaked into anomaly fitting")
    healthy = train_df[train_df["rul_true"] >= healthy_rul]
    if len(healthy) < 30:
        healthy = train_df
    values = healthy[columns].to_numpy(dtype=np.float64)
    mu = np.nanmean(values, axis=0)
    sigma = np.nanstd(values, axis=0)
    sigma = np.where(~np.isfinite(sigma) | (sigma < 1e-6), 1.0, sigma)
    scaled = _standardize(values, mu, sigma)
    sample = min(1024, len(scaled))
    forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=sample,
        contamination="auto",
        random_state=seed,
        n_jobs=n_jobs,
        bootstrap=False,
    )
    forest.fit(scaled)
    scores = -forest.decision_function(scaled)
    thresholds = {
        "q99": float(np.quantile(scores, 0.99)),
        "q995": float(np.quantile(scores, 0.995)),
        "q999": float(np.quantile(scores, 0.999)),
    }
    return AnomalyModel(forest=forest, mu=mu, sigma=sigma, columns=list(columns), score_thresholds=thresholds)


def score_anomaly(model: AnomalyModel, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return max absolute control-chart z and an isolation-forest score (higher is more anomalous)."""
    values = frame[model.columns].to_numpy(dtype=np.float64)
    scaled = _standardize(values, model.mu, model.sigma)
    control_z = np.max(np.abs(scaled), axis=1)
    iforest = -model.forest.decision_function(scaled)
    return control_z.astype(np.float64), iforest.astype(np.float64)
