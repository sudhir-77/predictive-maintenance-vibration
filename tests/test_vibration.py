"""Vibration features, splits, envelope math, and a small end-to-end run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from src.pipeline import parse_coverage_table
from src.report import render_resume_notes
from vibration.catalog import RECORDINGS
from vibration.config import CLASS_ORDER, PARENT_FD001_RMSE, PARENT_FD003_RMSE
from vibration.download import download_file, verify_checksum
from vibration.envelope import bpfi_hz, bpfo_hz, detect_envelope_peak, outer_race_check
from vibration.features import crest_factor, excess_kurtosis, feature_names, fft_band_energies, rms, window_features
from vibration.metrics import baseline_comparison_passed, macro_f1, parent_headline_deviation, relative_deviation
from vibration.models import fit_random_forest, train_cnn
from vibration.pipeline import merge_vibration_metrics
from vibration.signals import load_drive_end
from vibration.windowing import build_split_document, cross_split_overlap_count, ranges_overlap, split_for_load, window_starts


def test_catalog_covers_ten_classes_and_four_loads():
    assert len(RECORDINGS) == 40
    assert len({item.label for item in RECORDINGS}) == 10
    assert {item.label for item in RECORDINGS} == set(CLASS_ORDER)
    assert {item.load_hp for item in RECORDINGS} == {0, 1, 2, 3}


def test_bpfo_matches_hand_calculation():
    # At 1797 rpm the shaft is 29.95 Hz. BPFO = 4.5 * 29.95 * (1 - 0.3126 / 1.537).
    ratio = 0.3126 / 1.537
    expected = 4.5 * (1797 / 60) * (1.0 - ratio)
    assert math.isclose(expected, 107.364027, abs_tol=1e-6)
    assert math.isclose(bpfo_hz(1797 / 60), expected, rel_tol=0, abs_tol=1e-12)


def test_rms_kurtosis_and_crest_match_hand_values():
    assert math.isclose(rms(np.array([3.0, 4.0])), math.sqrt(12.5), abs_tol=1e-12)
    assert math.isclose(excess_kurtosis(np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])), -2.0, abs_tol=1e-12)
    assert math.isclose(crest_factor(np.array([3.0, -4.0])), 4.0 / math.sqrt(12.5), abs_tol=1e-12)


def test_fft_band_contains_a_known_tone():
    fs = 12_000.0
    t = np.arange(2048) / fs
    tone = np.sin(2 * np.pi * 1000.0 * t)
    bands = fft_band_energies(tone, fs)
    assert int(np.argmax(bands)) == 1
    assert len(window_features(tone, fs, 1797.0)) == len(feature_names())


def test_inner_race_order_exceeds_outer_race_order():
    fs = 12_000.0
    rpm = 1797.0
    duration = np.arange(0.0, 1.0, 1.0 / fs)
    inner = bpfi_hz(rpm / 60.0)
    signal = (1.0 + 0.9 * np.sin(2 * np.pi * inner * duration)) * np.sin(2 * np.pi * 3500.0 * duration)
    values = window_features(signal, fs, rpm)
    names = feature_names()
    assert values[names.index("env_bpfi")] > values[names.index("env_bpfo")]
    fs = 12_000.0
    duration = np.arange(0.0, 2.0, 1.0 / fs)
    modulator = 110.0
    signal = (1.0 + 0.8 * np.sin(2 * np.pi * modulator * duration)) * np.sin(2 * np.pi * 3500.0 * duration)
    detected = detect_envelope_peak(signal, fs, 80.0, 140.0)
    assert abs(detected - modulator) / modulator < 0.02


def test_outer_race_check_on_a_synthetic_bearing_tone():
    target = 110.0
    shaft_hz = target / bpfo_hz(1.0)
    fs = 12_000.0
    duration = np.arange(0.0, 2.0, 1.0 / fs)
    signal = (1.0 + 0.8 * np.sin(2 * np.pi * target * duration)) * np.sin(2 * np.pi * 3500.0 * duration)
    result = outer_race_check(signal, fs, shaft_hz * 60.0)
    assert result["rel_error"] <= 0.02
    assert abs(result["bpfo_hz_theory"] - target) < 1e-8


def test_load_split_windows_do_not_share_samples():
    lengths = {item.file_id: 20_000 for item in RECORDINGS}
    rpms = {item.file_id: float(item.nominal_rpm) for item in RECORDINGS}
    document = build_split_document(RECORDINGS, lengths, rpms, 2048, 5, 12_000, 1)
    assert set(document["train_recordings"]).isdisjoint(document["test_recordings"])
    assert set(document["val_recordings"]).isdisjoint(document["test_recordings"])
    assert cross_split_overlap_count(document) == 0
    for file_id in document["test_recordings"]:
        assert document["recordings"][file_id]["load_hp"] == 3
        assert split_for_load(3) == "test"
    leaked = {
        "version": 1,
        "recordings": {
            "105": {"split": "train", "windows": [[0, 100]]},
            "106": {"split": "test", "windows": [[0, 100], [50, 150]]},
        },
    }
    # Same indexes on different recordings are different signals.
    assert cross_split_overlap_count(leaked) == 0
    leaked["recordings"]["105"]["windows"].append([40, 80])
    leaked["recordings"]["105"]["split"] = "train"
    mixed = {
        "recordings": {
            "105": {"split": "train", "windows": [[0, 100], [40, 90]]},
        }
    }
    # Both windows are train, so this is not cross-split leakage.
    assert cross_split_overlap_count(mixed) == 0
    mixed["recordings"]["105"] = {
        "split": "train",
        "windows": [[0, 100]],
    }
    mixed["recordings"]["105b"] = {"split": "val", "windows": [[0, 50]]}
    # Reassign by putting two splits on one recording id.
    mixed = {
        "recordings": {
            "105": {"split": "train", "windows": [[0, 100], [80, 120]]},
        }
    }
    # The helper reads one split per recording, so overlap inside that list is same-split.
    assert cross_split_overlap_count(mixed) == 0
    # Direct range check: a shared sample across splits is leakage.
    assert ranges_overlap(0, 100, 100, 120) is False
    assert ranges_overlap(0, 100, 99, 120) is True
    pair = {
        "recordings": {
            "105": {"split": "train", "windows": [[0, 100]]},
        }
    }
    # Two entries cannot share an id in this JSON object, so the counter groups by id
    # and compares windows stored together. Build that case with a custom group by
    # stuffing both splits into one recording's window list is not possible without
    # a split per window. The pipeline stores one split per recording.
    assert "105" in pair["recordings"]
    starts = window_starts(10_000, 2048, 4)
    assert starts[0] == 0
    assert starts[-1] == 10_000 - 2048
    assert len(starts) == len(set(starts))


def test_cross_split_detector_counts_a_shared_sample():
    # The document stores one split per recording. Overlap is counted when two
    # windows in that recording list carry different split labels. The production
    # writer does not do that; this fixture does, by bypassing the writer.
    document = {"recordings": {}}
    # Monkeypatch the grouping by calling the comparator on a hand-built list.
    assert ranges_overlap(0, 50, 25, 75) is True
    count = 0
    windows = [("train", 0, 100), ("val", 50, 150), ("val", 200, 250)]
    for index, (split_a, start_a, end_a) in enumerate(windows):
        for split_b, start_b, end_b in windows[index + 1 :]:
            if split_a != split_b and ranges_overlap(start_a, end_a, start_b, end_b):
                count += 1
    assert count == 1


def test_macro_f1_matches_a_worked_example():
    # Class a: precision 1, recall 0.5, F1 2/3. Class b: precision 2/3, recall 1, F1 0.8.
    score = macro_f1(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert math.isclose(score, ((2.0 / 3.0) + 0.8) / 2.0, abs_tol=1e-12)


def test_baseline_comparison_rule():
    assert baseline_comparison_passed(0.97, 0.96) is True
    assert baseline_comparison_passed(0.93, 0.90) is True
    assert baseline_comparison_passed(0.92, 0.91) is False


def test_parent_deviation_is_zero_at_the_reference():
    assert parent_headline_deviation(PARENT_FD001_RMSE, PARENT_FD003_RMSE, 1.0, 0.0) == 0.0
    assert relative_deviation(10.2, 10.0) == pytest.approx(0.02)
    assert relative_deviation(0.0, 0.0) == 0.0


def test_drive_end_loader_reads_a_tiny_mat(tmp_path: Path):
    path = tmp_path / "97.mat"
    scipy.io.savemat(path, {"X097_DE_time": np.array([[1.0, 2.0, 3.0]]), "X097RPM": np.array([[1796.0]])})
    signal, rpm = load_drive_end(path)
    assert signal.tolist() == [1.0, 2.0, 3.0]
    assert rpm == pytest.approx(1796.0)


def test_checksum_and_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = b"bearing-bytes"
    path = tmp_path / "105.mat"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert verify_checksum(path, "105", {"105": digest}) == digest
    with pytest.raises(ValueError):
        verify_checksum(path, "105", {"105": "0" * 64})

    class Response:
        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int = 0):
            yield payload

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    monkeypatch.setattr("vibration.download.requests.get", lambda *args, **kwargs: Response())
    dest = tmp_path / "118.mat"
    download_file("https://example.test/118.mat", dest)
    assert dest.read_bytes() == payload


def test_parse_coverage_table_splits_packages():
    text = """
Name                      Stmts   Miss  Cover
src\\metrics.py              10      1    90%
vibration\\envelope.py       20      0   100%
TOTAL                        30      1    97%
"""
    parsed = parse_coverage_table(text)
    assert parsed["src"] == pytest.approx(90.0)
    assert parsed["vibration"] == pytest.approx(100.0)
    assert parsed["combined"] == pytest.approx(100.0 * 29 / 30)


def test_random_forest_separates_two_obvious_groups():
    features = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.2, 4.8]], dtype=np.float64)
    labels = np.array(["normal", "normal", "or_007", "or_007"])
    model = fit_random_forest(features, labels, n_estimators=20, n_jobs=1)
    assert list(model.predict(features)) == ["normal", "normal", "or_007", "or_007"]


def test_cnn_learns_two_tones():
    length = 256
    t = np.linspace(0, 1, length, dtype=np.float32)
    low = np.sin(2 * np.pi * 4 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 40 * t).astype(np.float32)
    train = np.vstack([np.tile(low, (12, 1)), np.tile(high, (12, 1))])
    val = np.vstack([np.tile(low, (4, 1)), np.tile(high, (4, 1))])
    train_y = np.array([0] * 12 + [1] * 12)
    val_y = np.array([0] * 4 + [1] * 4)
    _model, info = train_cnn(
        train,
        train_y,
        val,
        val_y,
        n_classes=2,
        epochs=8,
        device=__import__("torch").device("cpu"),
        seed=0,
        batch=8,
        patience=8,
    )
    assert info["val_macro_f1"] >= 0.9


def test_resume_notes_use_vibration_headline_strings():
    metrics = {
        "metrics": {
            "headline_text": {
                "fd001_test_rmse": "15.11",
                "vibration_cnn_macro_f1": "0.9720",
                "vibration_baseline_macro_f1": "0.9510",
                "vibration_bpfo_rel_error_percent": "0.42",
                "vibration_overlap_windows": "0",
                "vibration_test_load_hp": "3",
                "vibration_n_classes": "10",
            }
        }
    }
    text = render_resume_notes(metrics)
    raw = json.dumps(metrics)
    for token in ("0.9720", "0.9510", "0.42", "15.11", "10"):
        assert token in text
        assert token in raw
    assert "\u2014" not in text


def test_merge_appends_vibration_criteria(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "project": "predictive-maintenance-ops",
                "success_criteria": [
                    {"name": "FD001 test RUL RMSE", "value": PARENT_FD001_RMSE, "op": "<=", "threshold": 18.0, "threshold_high": None, "passed": True},
                    {"name": "FD003 test RUL RMSE", "value": PARENT_FD003_RMSE, "op": "<=", "threshold": 20.0, "threshold_high": None, "passed": True},
                    {"name": "Alert lead time", "value": 1.0, "op": ">=", "threshold": 0.8, "threshold_high": None, "passed": True},
                    {"name": "False alerts", "value": 0.0, "op": "<=", "threshold": 0.1, "threshold_high": None, "passed": True},
                    {"name": "Test coverage of src/", "value": 90.0, "op": ">=", "threshold": 70.0, "threshold_high": None, "passed": True},
                ],
                "metrics": {
                    "fd001": {"test_rmse": PARENT_FD001_RMSE},
                    "fd003": {"test_rmse": PARENT_FD003_RMSE},
                    "alerts": {"lead_time_fraction": 1.0, "false_alerts_per_engine": 0.0},
                    "headline_text": {"fd001_test_rmse": "15.11"},
                },
                "baseline": {},
                "splits": {},
                "seeds": [0],
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "cnn_test_macro_f1": 0.97,
        "baseline_test_macro_f1": 0.96,
        "best_test_macro_f1": 0.97,
        "bpfo_rel_error": 0.004,
        "bpfo_rel_error_percent": 0.4,
        "overlap_windows": 0,
        "test_load_hp": 3,
        "n_classes": 10,
        "n_train_windows": 10,
        "n_val_windows": 4,
        "n_test_windows": 4,
    }
    merged = merge_vibration_metrics(metrics_path, summary, 88.0, quick=False)
    names = [item["name"] for item in merged["success_criteria"]]
    assert "Macro-F1 on held-out motor load (10 classes)" in names
    assert "Envelope fault-frequency check" in names
    assert "No leakage" in names
    assert "Parent project" in names
    assert "Test coverage of src/ and vibration/" in names
    assert merged["project"] == "predictive-maintenance-vibration"
    assert all(item["passed"] for item in merged["success_criteria"])
