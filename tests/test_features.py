"""SQL features, regimes, and train-only normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import assign_regimes, build_feature_table, fit_regime_stats


def _frame(n: int, unit: int, origin: str, sensor_value, op1: float = 0.0) -> pd.DataFrame:
    cycle = np.arange(1, n + 1)
    if callable(sensor_value):
        s2 = sensor_value(cycle).astype(np.float64)
    else:
        s2 = np.full(n, sensor_value, dtype=np.float64)
    return pd.DataFrame(
        {
            "dataset": "FD001",
            "origin": origin,
            "split": "train" if origin == "train_file" else "test",
            "engine_key": f"{origin}-{unit}",
            "unit_id": unit,
            "cycle": cycle.astype(np.int32),
            "source_row": np.arange(1, n + 1),
            "rul_true": (n - cycle).astype(np.float64),
            "op1": op1,
            "op2": 0.0,
            "op3": 100.0,
            "s2": s2,
        }
    )


def test_regime_collapses_sea_level_noise_and_keeps_real_shifts():
    frame = pd.DataFrame(
        {
            "op1": [-0.008, 0.008, 20.0],
            "op2": [0.0004, -0.0004, 0.25],
            "op3": [100.0, 100.2, 20.0],
        }
    )
    labels = assign_regimes(frame).tolist()
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
    assert labels[2].startswith("20|")


def test_rolling_slope_matches_polyfit_on_z_scores():
    train = _frame(12, 1, "train_file", lambda cycle: cycle.astype(np.float64))
    mask = pd.Series(True, index=train.index)
    featured, sensors, _signs = build_feature_table(train, mask, sensors=["s2"])
    assert sensors == ["s2"]
    z_values = featured["s2_z"].to_numpy(dtype=np.float64)
    cycles = featured["cycle"].to_numpy(dtype=np.float64)
    expected = float(np.polyfit(cycles[-10:], z_values[-10:], 1)[0])
    assert abs(float(featured["s2_slope_10"].iloc[-1]) - expected) < 1e-4
    # Partial windows are filled, not left as NaN.
    assert np.isfinite(featured["s2_slope_10"]).all()


def test_regime_stats_ignore_validation_rows():
    train = _frame(12, 1, "train_file", lambda cycle: cycle.astype(np.float64))
    train["split"] = "train"
    val = _frame(12, 2, "train_file", 1000.0)
    val["split"] = "val"
    val["engine_key"] = "train_file-2"
    frame = pd.concat([train, val], ignore_index=True)
    mask = frame["split"].eq("train")
    stats = fit_regime_stats(frame.loc[mask].assign(regime="0|0.00|100"), ["s2"])
    assert abs(float(stats["s2__mean"].iloc[0]) - float(train["s2"].mean())) < 1e-6
    featured, _, _ = build_feature_table(frame, mask, sensors=["s2"])
    train_z = featured.loc[featured["engine_key"] == "train_file-1", "s2_z"]
    val_z = featured.loc[featured["engine_key"] == "train_file-2", "s2_z"]
    assert float(train_z.abs().mean()) < 2
    assert float(val_z.mean()) > 50


def test_feature_names_do_not_include_labels():
    train = _frame(12, 1, "train_file", lambda cycle: cycle.astype(np.float64))
    featured, sensors, _ = build_feature_table(train, pd.Series(True, index=train.index), sensors=["s2"])
    from src.features import tabular_feature_names

    columns = tabular_feature_names(sensors)
    assert not any("rul" in column for column in columns)
    assert "health_index" in featured.columns
    assert featured["health_index"].notna().all()


def test_fault_mode_separates_two_training_endings():
    parts = []
    for unit in range(1, 5):
        rising = _frame(20, unit, "train_file", lambda cycle: cycle.astype(np.float64))
        rising["split"] = "train"
        parts.append(rising)
    for unit in range(5, 9):
        falling = _frame(20, unit, "train_file", lambda cycle: -cycle.astype(np.float64))
        falling["split"] = "train"
        parts.append(falling)
    frame = pd.concat(parts, ignore_index=True)
    featured, _, _ = build_feature_table(frame, frame["split"].eq("train"), sensors=["s2"])
    last = featured.sort_values("cycle").groupby("unit_id", sort=False).tail(1)
    rising_score = float(last.loc[last["unit_id"] < 5, "fault_mode"].mean())
    falling_score = float(last.loc[last["unit_id"] >= 5, "fault_mode"].mean())
    assert rising_score * falling_score < 0
    early = float(featured.loc[featured["unit_id"] == 1].sort_values("cycle")["fault_mode"].iloc[0])
    late = float(featured.loc[featured["unit_id"] == 1].sort_values("cycle")["fault_mode"].iloc[-1])
    assert abs(late) > abs(early)
