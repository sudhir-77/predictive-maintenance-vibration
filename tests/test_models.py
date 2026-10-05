"""Splits, calibrator, baselines, anomaly scores, and small model fits."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.anomaly import fit_anomaly, score_anomaly
from src.baseline import fit_piecewise, fit_sensor_threshold, predict_piecewise, sensor_threshold_alerts
from src.metrics import rmse
from src.selection import choose_alpha, choose_calibrator, maybe_calibrate
from src.splits import assign_units, load_or_create_split


def test_assign_units_is_disjoint_and_reproducible():
    train, val = assign_units(list(range(1, 21)), 0.2, seed=0)
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(1, 21))
    assert (train, val) == assign_units(list(range(1, 21)), 0.2, seed=0)


def test_saved_split_is_reused(tmp_path):
    path = tmp_path / "FD001.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset": "FD001",
                "quick": False,
                "seed": 0,
                "val_fraction": 0.2,
                "train_units": [1, 2, 3],
                "val_units": [4],
                "test_units": [10, 11],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_or_create_split(path, "FD001", [1, 2, 3, 4], [10, 11], 0.2, seed=1, quick=False)
    assert loaded["train_units"] == [1, 2, 3]
    assert loaded["val_units"] == [4]


def test_calibrator_removes_a_constant_bias_and_blend_prefers_the_accurate_model():
    truth = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    biased = truth - 8.0
    (intercept, slope), calibrated = maybe_calibrate(truth, biased)
    assert abs(intercept - 8.0) < 1e-6
    assert abs(slope - 1.0) < 1e-6
    assert abs(float(np.max(np.abs(calibrated - truth)))) < 1e-6
    perfect = truth.copy()
    wrong = np.full_like(truth, 3.0)
    alpha, _score = choose_alpha(truth, perfect, wrong)
    assert alpha == 1.0
    alpha_cnn, _score = choose_alpha(truth, wrong, perfect)
    assert alpha_cnn == 0.0


def test_isotonic_is_kept_for_a_curve_and_a_line_stays_linear():
    truth = np.tile(np.linspace(10.0, 80.0, 40), 4)
    groups = np.repeat(np.array(["a", "b", "c", "d"]), 40)
    linear_cal, linear_hat = choose_calibrator(truth, truth - 8.0, groups)
    assert linear_cal.kind == "linear"
    assert abs(linear_cal.intercept - 8.0) < 1e-6
    assert rmse(truth, linear_hat) < 1e-6
    curved = np.sqrt(np.clip(truth, 0, None)) * 8.0
    iso_cal, iso_hat = choose_calibrator(truth, curved, groups)
    assert iso_cal.kind == "isotonic"
    assert rmse(truth, iso_hat) + 1.0 < rmse(truth, curved)


def test_piecewise_baseline_recovers_a_line_and_clips():
    frame = pd.DataFrame(
        {
            "s2_delta": np.linspace(-2, 2, 20),
            "s2_roll_mean_10": np.linspace(0, 5, 20),
            "rul_target": np.linspace(100, 0, 20),
            "rul_true": np.linspace(100, 0, 20),
            "engine_key": ["e"] * 20,
            "cycle": np.arange(20),
        }
    )
    # High sensor, low RUL, so correlation is negative and the late threshold alerts on high values.
    frame["rul_target"] = 50 - 10 * frame["s2_delta"]
    frame["rul_true"] = frame["rul_target"]
    model = fit_piecewise(frame, ["s2"])
    pred = predict_piecewise(model, frame)
    assert pred.min() >= 0
    assert pred.max() <= model.clip
    rule = fit_sensor_threshold(frame, ["s2"])
    assert rule.direction == "high"
    alerts = sensor_threshold_alerts(frame, rule)
    assert len(alerts) == len(frame)
    assert rule.sensor == "s2"


def test_control_chart_flags_a_sustained_shift():
    rng = np.random.default_rng(0)
    healthy = rng.normal(0, 1, size=(80, 2))
    rows = []
    for i in range(80):
        rows.append(_anomaly_row(i, healthy[i], rul=100, split="train"))
    shifted = np.array([12.0, 12.0])
    rows.append(_anomaly_row(80, shifted, rul=5, split="val"))
    frame = pd.DataFrame(rows)
    model = fit_anomaly(frame[frame["split"] == "train"], ["s2_roll_mean_10", "s4_roll_mean_10"], n_estimators=50, n_jobs=1)
    z, scores = score_anomaly(model, frame)
    assert z[-1] > 5
    assert scores[-1] > np.median(scores[:-1])


def test_lightgbm_fits_a_line():
    from src.lgbm_model import predict_lgbm, train_lgbm

    x_values = np.linspace(0, 10, 120)
    frame = pd.DataFrame(
        {
            "cycle_feat": x_values,
            "rul_true": 3 * x_values,
            "rul_target": 3 * x_values,
            "origin": "train_file",
            "split": "train",
        }
    )
    model = train_lgbm(frame, ["cycle_feat"], fixed_rounds=60, seed=0, min_child_samples=5)
    pred = predict_lgbm(model, frame)
    assert np.isfinite(pred).all()
    assert float(np.mean((pred - frame["rul_target"]) ** 2)) < 10.0


def test_cnn_outputs_finite_predictions():
    from src.cnn_model import predict_cnn, resolve_device, train_cnn

    rows = []
    for unit in (1, 2):
        for cycle in range(1, 9):
            rows.append(
                {
                    "engine_key": f"train_file-{unit}",
                    "unit_id": unit,
                    "cycle": cycle,
                    "origin": "train_file",
                    "split": "train",
                    "s2_z": float(cycle) / 10.0,
                    "rul_target": float(20 - cycle),
                    "rul_true": float(20 - cycle),
                }
            )
    frame = pd.DataFrame(rows)
    device = resolve_device("cpu")
    model, epochs = train_cnn(frame, ["s2_z"], device, val_df=None, window=4, epochs=2, patience=1, batch_size=4, seed=0)
    assert epochs >= 1
    pred = predict_cnn(model, frame, ["s2_z"], device, window=4)
    assert pred.shape == (len(frame),)
    assert np.isfinite(pred).all()


def _anomaly_row(index: int, values: np.ndarray, rul: float, split: str) -> dict:
    return {
        "origin": "train_file",
        "split": split,
        "rul_true": rul,
        "s2_roll_mean_10": float(values[0]),
        "s4_roll_mean_10": float(values[1]),
        "cycle": index + 1,
    }
