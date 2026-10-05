"""DuckDB SQL features: regime z-scores, rolling means, and rolling slopes."""

from __future__ import annotations

import re

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import (
    N_THREADS,
    OP_COLS,
    ROLL_MEAN_WINDOWS,
    ROLL_STD_WINDOWS,
    RUL_CLIP,
    SEED,
    SENSOR_COLS,
    SENSOR_STD_MIN,
    SLOPE_WINDOWS,
)

_SENSOR_NAME = re.compile(r"^s\d+$")


def _check_sensor(name: str) -> str:
    if not _SENSOR_NAME.match(name):
        raise ValueError(f"unexpected sensor name {name}")
    return name


def assign_regimes(frame: pd.DataFrame) -> pd.Series:
    """Discretize operating settings.

    FD001 and FD003 jitter around a single sea-level condition. Rounding collapses
    that jitter into one regime. Settings that differ by whole units stay apart.
    """
    op1 = np.rint(frame["op1"].to_numpy(dtype=np.float64)).astype(np.int32)
    op2 = np.round(frame["op2"].to_numpy(dtype=np.float64), 2)
    op3 = np.rint(frame["op3"].to_numpy(dtype=np.float64)).astype(np.int32)
    labels = []
    for a, b, c in zip(op1, op2, op3):
        op2_text = "0.00" if abs(float(b)) < 1e-8 else f"{float(b):.2f}"
        labels.append(f"{int(a)}|{op2_text}|{int(c)}")
    return pd.Series(labels, index=frame.index, name="regime")


def informative_sensors(train: pd.DataFrame, sensors: list[str], min_std: float = SENSOR_STD_MIN) -> list[str]:
    kept: list[str] = []
    for sensor in sensors:
        _check_sensor(sensor)
        if sensor not in train.columns:
            continue
        std = float(np.nanstd(train[sensor].to_numpy(dtype=np.float64)))
        if std >= min_std:
            kept.append(sensor)
    if not kept:
        raise ValueError("no sensors passed the variance filter")
    return kept


def fit_regime_stats(train: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Mean and population std of each sensor inside each operating regime, from train rows only."""
    rows: list[dict] = []
    for regime, group in train.groupby("regime", sort=False):
        row: dict = {"regime": regime}
        for sensor in sensors:
            values = group[sensor].to_numpy(dtype=np.float64)
            mean = float(np.mean(values))
            std = float(np.std(values))
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            row[f"{sensor}__mean"] = mean
            row[f"{sensor}__std"] = std
        rows.append(row)
    if not rows:
        raise ValueError("cannot fit regime stats on an empty train set")
    return pd.DataFrame(rows)


def tabular_feature_names(sensors: list[str]) -> list[str]:
    columns = ["cycle_feat", "health_index", "health_max", "fault_mode"]
    for sensor in sensors:
        columns.extend(
            [
                f"{sensor}_z",
                f"{sensor}_delta",
                f"{sensor}_resid",
                f"{sensor}_accel",
                *[f"{sensor}_roll_mean_{window}" for window in ROLL_MEAN_WINDOWS],
                *[f"{sensor}_roll_std_{window}" for window in ROLL_STD_WINDOWS],
                *[f"{sensor}_slope_{window}" for window in SLOPE_WINDOWS],
            ]
        )
    return columns


def cnn_feature_names(sensors: list[str]) -> list[str]:
    return [f"{sensor}_z" for sensor in sensors]


def _window_sql(sensors: list[str]) -> tuple[str, str]:
    windows = sorted(set(ROLL_MEAN_WINDOWS) | set(ROLL_STD_WINDOWS) | set(SLOPE_WINDOWS))
    clauses = []
    for window in windows:
        clauses.append(
            f"w{window} AS (PARTITION BY engine_key ORDER BY cycle "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW)"
        )
    exprs: list[str] = []
    for sensor in sensors:
        for window in ROLL_MEAN_WINDOWS:
            exprs.append(f"AVG({sensor}_z) OVER w{window} AS {sensor}_roll_mean_{window}")
        for window in ROLL_STD_WINDOWS:
            exprs.append(f"STDDEV_SAMP({sensor}_z) OVER w{window} AS {sensor}_roll_std_{window}")
        for window in SLOPE_WINDOWS:
            exprs.append(
                f"CASE WHEN COUNT(*) OVER w{window} >= {window} THEN "
                f"(AVG(CAST(cycle AS DOUBLE) * {sensor}_z) OVER w{window} "
                f"- AVG(CAST(cycle AS DOUBLE)) OVER w{window} * AVG({sensor}_z) OVER w{window}) "
                f"/ NULLIF(VAR_POP(CAST(cycle AS DOUBLE)) OVER w{window}, 0) "
                f"ELSE NULL END AS {sensor}_slope_{window}"
            )
    return ",\n  ".join(exprs), ",\n  ".join(clauses)


def sql_normalize_and_roll(frame: pd.DataFrame, stats: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Z-score sensors inside the train-fit regime, then compute rolling means and slopes."""
    for sensor in sensors:
        _check_sensor(sensor)
    z_exprs = [
        f"COALESCE((t.{sensor} - st.{sensor}__mean) / NULLIF(st.{sensor}__std, 0), 0) AS {sensor}_z"
        for sensor in sensors
    ]
    roll_exprs, window_clause = _window_sql(sensors)
    query = f"""
    WITH scored AS (
      SELECT
        t.dataset, t.origin, t.split, t.is_train, t.engine_key, t.unit_id, t.cycle, t.source_row,
        t.regime, t.op1, t.op2, t.op3, t.rul_true,
        {", ".join(z_exprs)}
      FROM telemetry t
      LEFT JOIN regime_stats st ON t.regime = st.regime
    )
    SELECT
      *,
      {roll_exprs}
    FROM scored
    WINDOW
      {window_clause}
    """
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads={N_THREADS}")
        connection.execute("SET memory_limit='1GB'")
        connection.register("telemetry", frame)
        connection.register("regime_stats", stats)
        featured = connection.execute(query).df()
    finally:
        connection.close()
    return featured


def degradation_signs(train: pd.DataFrame, sensors: list[str], target: str = "rul_target") -> dict[str, float]:
    """Positive sign means the z-scored sensor rises as remaining life falls."""
    signs: dict[str, float] = {}
    for sensor in sensors:
        column = f"{sensor}_z"
        corr = train[column].corr(train[target])
        if corr is None or not np.isfinite(corr) or abs(float(corr)) < 1e-8:
            signs[sensor] = 0.0
        else:
            signs[sensor] = float(-np.sign(corr))
    return signs


def apply_health_index(frame: pd.DataFrame, signs: dict[str, float]) -> pd.DataFrame:
    used = [sensor for sensor, sign in signs.items() if sign != 0.0]
    if not used:
        frame["health_index"] = np.float32(0.0)
        frame["health_max"] = np.float32(0.0)
        return frame
    signed = np.vstack(
        [signs[sensor] * frame[f"{sensor}_z"].to_numpy(dtype=np.float64) for sensor in used]
    )
    frame["health_index"] = signed.mean(axis=0).astype(np.float32)
    # Worst sensor, not the average. FD003's two fault modes move different sensors.
    frame["health_max"] = signed.max(axis=0).astype(np.float32)
    return frame


def fit_fault_modes(train: pd.DataFrame, sensors: list[str]) -> dict | None:
    """Two end-of-life sensor-delta centroids from training engines only.

    FD003 mixes two fault modes that move different sensors. The score is applied
    later from the current cycle's deltas, so a unit's future cycles are not used.
    """
    last = train.sort_values("cycle").groupby("engine_key", sort=False).tail(1)
    if last["engine_key"].nunique() < 4:
        return None
    columns = [f"{sensor}_delta" for sensor in sensors]
    values = last[columns].to_numpy(dtype=np.float64)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    standardized = (values - center) / scale
    if not np.isfinite(standardized).all():
        return None
    cluster = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    cluster.fit(standardized)
    return {
        "center": center,
        "scale": scale,
        "centroids": cluster.cluster_centers_.astype(np.float64),
    }


def apply_fault_mode(frame: pd.DataFrame, sensors: list[str], model: dict | None) -> pd.DataFrame:
    """Signed closeness to the two training fault centroids. Zero when no model was fit."""
    if model is None:
        frame["fault_mode"] = np.float32(0.0)
        return frame
    columns = [f"{sensor}_delta" for sensor in sensors]
    standardized = (frame[columns].to_numpy(dtype=np.float64) - model["center"]) / model["scale"]
    distance_a = np.linalg.norm(standardized - model["centroids"][0], axis=1)
    distance_b = np.linalg.norm(standardized - model["centroids"][1], axis=1)
    score = (distance_a - distance_b) / (distance_a + distance_b + 1e-8)
    frame["fault_mode"] = score.astype(np.float32)
    return frame


def _prepare_input(frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    required = [
        "dataset",
        "origin",
        "split",
        "is_train",
        "engine_key",
        "unit_id",
        "cycle",
        "source_row",
        "rul_true",
        *OP_COLS,
        *sensors,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"feature frame is missing {missing}")
    out = frame.copy()
    if "regime" not in out.columns:
        out["regime"] = assign_regimes(out)
    out["rul_true"] = out["rul_true"].astype(np.float64)
    for sensor in sensors:
        out[sensor] = out[sensor].astype(np.float64)
    for column in OP_COLS:
        out[column] = out[column].astype(np.float64)
    return out


def build_feature_table(
    frame: pd.DataFrame,
    train_mask: pd.Series,
    sensors: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Build model features. Regime statistics and health-index signs use train rows only."""
    if isinstance(train_mask, pd.Series):
        mask = train_mask.reindex(frame.index).fillna(False).astype(bool)
    else:
        mask = pd.Series(np.asarray(train_mask, dtype=bool), index=frame.index)
    if sensors is None:
        chosen = informative_sensors(frame.loc[mask], [col for col in SENSOR_COLS if col in frame.columns])
    else:
        chosen = [_check_sensor(sensor) for sensor in sensors]
    prepared = frame.copy()
    prepared["is_train"] = mask.to_numpy().astype(np.int8)
    prepared = _prepare_input(prepared, chosen)
    prepared["regime"] = assign_regimes(prepared)
    train_rows = prepared.loc[mask.to_numpy()]
    if train_rows.empty:
        raise ValueError("train mask selects no rows")
    stats = fit_regime_stats(train_rows, chosen)
    fallback = str(stats["regime"].iloc[0])
    prepared.loc[~prepared["regime"].isin(set(stats["regime"])), "regime"] = fallback
    featured = sql_normalize_and_roll(prepared, stats, chosen)
    featured = featured.sort_values(["engine_key", "cycle"]).reset_index(drop=True)
    for sensor in chosen:
        z_values = featured[f"{sensor}_z"]
        first = featured.groupby("engine_key", sort=False)[f"{sensor}_z"].transform("first")
        featured[f"{sensor}_delta"] = z_values - first
        featured[f"{sensor}_accel"] = featured[f"{sensor}_roll_mean_5"] - featured[f"{sensor}_roll_mean_20"]
        featured[f"{sensor}_resid"] = z_values - featured[f"{sensor}_roll_mean_20"]
    featured["rul_target"] = np.where(
        featured["rul_true"].notna(),
        np.minimum(featured["rul_true"].to_numpy(dtype=np.float64), RUL_CLIP),
        np.nan,
    )
    train_index = featured["is_train"].astype(bool)
    if not train_index.any():
        raise ValueError("train mask selects no rows after the SQL feature build")
    signs = degradation_signs(featured.loc[train_index], chosen)
    featured = apply_health_index(featured, signs)
    featured = apply_fault_mode(featured, chosen, fit_fault_modes(featured.loc[train_index], chosen))
    featured["cycle_feat"] = featured["cycle"].astype(np.float32)
    feature_cols = tabular_feature_names(chosen)
    block = featured[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    featured[feature_cols] = block.astype(np.float32)
    featured.attrs["signs"] = signs
    return featured, chosen, signs
