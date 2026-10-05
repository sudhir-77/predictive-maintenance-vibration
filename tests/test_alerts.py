"""Alert rules and the validation-only tuner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.alerts import AlertPolicy, apply_policy, persist_true, rolling_median, tune_policy, unit_alerts


def test_persist_true_waits_for_a_full_run():
    flag = np.array([1, 1, 1, 0, 0], dtype=bool)
    out = persist_true(flag, 3)
    assert out.tolist() == [False, False, True, False, False]
    assert persist_true(flag, 1).tolist() == [True, True, True, False, False]


def test_rolling_median_is_causal():
    values = np.array([1.0, 100.0, 3.0])
    smooth = rolling_median(values, 3)
    assert smooth[0] == 1.0
    assert smooth[-1] == 3.0


def test_policy_does_not_read_true_rul():
    frame = _engine(pred=np.array([80.0, 40.0, 20.0, 10.0]), rul=np.array([80.0, 40.0, 20.0, 10.0]))
    policy = AlertPolicy(30, 1, 1, 1e12, 1e12, 0, 1)
    first = apply_policy(frame, policy).tolist()
    frame = frame.copy()
    frame["rul_true"] = [1.0, 1.0, 1.0, 1.0]
    second = apply_policy(frame, policy).tolist()
    assert first == second
    assert first == [False, False, True, True]


def test_tuner_finds_a_policy_on_a_perfect_forecast():
    # True RUL falls by 1 each cycle. A threshold of 30 alerts inside the safe window.
    parts = []
    for unit in range(4):
        rul = np.arange(80, -1, -1, dtype=float)
        parts.append(_engine(pred=rul.copy(), rul=rul, unit=unit))
    frame = pd.concat(parts, ignore_index=True)
    good = AlertPolicy(30, 1, 1, 1e12, 1e12, 0, 1)
    bad = AlertPolicy(90, 1, 1, 1e12, 1e12, 0, 1)
    policy, stats = tune_policy(frame, {"q99": 0.0, "q999": 0.0}, policies=[bad, good])
    assert policy.rul_threshold == 30
    assert stats["passed_constraints"] is True
    assert stats["lead_time_fraction"] == 1.0
    assert stats["false_alerts_per_engine"] == 0.0


def test_tuner_refuses_test_rows():
    frame = _engine(pred=np.array([10.0]), rul=np.array([10.0]))
    frame["split"] = "test"
    frame["origin"] = "test_file"
    try:
        tune_policy(frame, {"q99": 1.0}, policies=[AlertPolicy(30, 1, 1, 1e12, 1e12, 0, 1)])
    except RuntimeError as exc:
        assert "test" in str(exc)
    else:
        raise AssertionError("expected a test-set refusal")


def test_gated_anomaly_can_alert_when_rul_forecast_is_only_somewhat_low():
    pred = np.array([40.0, 40.0, 40.0])
    z = np.array([0.0, 9.0, 9.0])
    iforest = np.array([0.0, 0.0, 0.0])
    policy = AlertPolicy(rul_threshold=30, persist_cycles=2, smooth_window=1, z_threshold=8, iforest_threshold=1e12, anomaly_margin=15, anomaly_persist=2)
    # Forecast stays at 40, so the RUL rule (threshold 30) is off.
    # Anomaly persists and 40 < 30 + 15, so the gated anomaly rule alerts.
    flags = unit_alerts(pred, z, iforest, policy)
    assert flags.tolist() == [False, False, True]


def _engine(pred: np.ndarray, rul: np.ndarray, unit: int = 1) -> pd.DataFrame:
    cycle = np.arange(1, len(pred) + 1)
    return pd.DataFrame(
        {
            "dataset": "FD001",
            "origin": "train_file",
            "split": "val",
            "engine_key": f"train_file-{unit}",
            "unit_id": unit,
            "cycle": cycle,
            "rul_true": rul,
            "rul_pred": pred,
            "control_chart_z": np.zeros(len(pred)),
            "iforest_score": np.zeros(len(pred)),
        }
    )
