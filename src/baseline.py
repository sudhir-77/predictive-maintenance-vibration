"""Constant mean RUL, a clipped linear sensor model, and a single-sensor alert."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.alerts import persist_true
from src.config import RUL_CLIP


@dataclass
class PiecewiseBaseline:
    sensor: str
    intercept: float
    coef: float
    clip: float
    constant: float


def fit_piecewise(train_df: pd.DataFrame, sensors: list[str], clip: float = RUL_CLIP) -> PiecewiseBaseline:
    """Linear map from the sensor delta most tied to clipped RUL, then clip to [0, clip]."""
    best_sensor = sensors[0]
    best_corr = -1.0
    for sensor in sensors:
        column = f"{sensor}_delta"
        corr = train_df[column].corr(train_df["rul_target"])
        if corr is not None and np.isfinite(corr) and abs(float(corr)) > best_corr:
            best_corr = abs(float(corr))
            best_sensor = sensor
    x_values = train_df[f"{best_sensor}_delta"].to_numpy(dtype=np.float64)
    y_values = train_df["rul_target"].to_numpy(dtype=np.float64)
    constant = float(np.mean(y_values))
    var = float(np.var(x_values))
    if var < 1e-12:
        intercept, coef = constant, 0.0
    else:
        coef = float(np.cov(x_values, y_values, ddof=0)[0, 1] / var)
        intercept = float(np.mean(y_values) - coef * np.mean(x_values))
    return PiecewiseBaseline(best_sensor, intercept, coef, clip, constant)


def predict_piecewise(model: PiecewiseBaseline, frame: pd.DataFrame) -> np.ndarray:
    x_values = frame[f"{model.sensor}_delta"].to_numpy(dtype=np.float64)
    pred = model.intercept + model.coef * x_values
    return np.clip(pred, 0.0, model.clip)


def predict_constant(model: PiecewiseBaseline, n_rows: int) -> np.ndarray:
    return np.full(n_rows, model.constant, dtype=np.float64)


@dataclass
class SensorThreshold:
    sensor: str
    column: str
    threshold: float
    direction: str
    persist: int = 5


def fit_sensor_threshold(train_df: pd.DataFrame, sensors: list[str]) -> SensorThreshold:
    """Alert when the most degraded sensor reaches its typical value 20 to 40 cycles before failure."""
    best_sensor = sensors[0]
    best_corr = -1.0
    best_column = f"{sensors[0]}_roll_mean_10"
    for sensor in sensors:
        column = f"{sensor}_roll_mean_10"
        corr = train_df[column].corr(train_df["rul_target"])
        if corr is not None and np.isfinite(corr) and abs(float(corr)) > best_corr:
            best_corr = abs(float(corr))
            best_sensor = sensor
            best_column = column
    corr = train_df[best_column].corr(train_df["rul_target"])
    direction = "high" if corr is not None and float(corr) < 0 else "low"
    late = train_df[train_df["rul_true"].between(20, 40)]
    if late.empty:
        threshold = float(train_df[best_column].quantile(0.8 if direction == "high" else 0.2))
    else:
        threshold = float(late[best_column].median())
    return SensorThreshold(best_sensor, best_column, threshold, direction)


def sensor_threshold_alerts(frame: pd.DataFrame, rule: SensorThreshold) -> pd.Series:
    alerts = pd.Series(False, index=frame.index)
    for _, group in frame.groupby("engine_key", sort=False):
        ordered = group.sort_values("cycle")
        values = ordered[rule.column].to_numpy(dtype=np.float64)
        if rule.direction == "high":
            hit = values >= rule.threshold
        else:
            hit = values <= rule.threshold
        alerts.loc[ordered.index] = persist_true(hit, rule.persist)
    return alerts
