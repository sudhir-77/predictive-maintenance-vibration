"""Quick vibration path. Uses cached CWRU files when they are already downloaded."""

from __future__ import annotations

import json
import time
from pathlib import Path

from vibration.pipeline import run_vibration


def test_quick_vibration_writes_artifacts(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    summary = run_vibration(root=root, quick=True, device="cpu", output_dir=tmp_path)
    assert time.perf_counter() - started < 120
    predictions = (tmp_path / "vibration_test_predictions.csv").read_text(encoding="utf-8").splitlines()
    header = predictions[0].split(",")
    assert header == ["window_id", "recording", "load_hp", "label", "cnn_pred", "baseline_pred"]
    assert len(predictions) > 1
    document = json.loads((tmp_path / "vibration_splits.json").read_text(encoding="utf-8"))
    for key in ("train_recordings", "val_recordings", "test_recordings"):
        assert key in document
    assert set(document["train_recordings"]).isdisjoint(document["val_recordings"])
    assert set(document["train_recordings"]).isdisjoint(document["test_recordings"])
    assert set(document["val_recordings"]).isdisjoint(document["test_recordings"])
    envelope = json.loads((tmp_path / "envelope_check.json").read_text(encoding="utf-8"))
    for key in ("bpfo_hz_theory", "bpfo_hz_detected", "rel_error"):
        assert key in envelope
    assert envelope["rel_error"] <= 0.02
    assert summary["overlap_windows"] == 0
    assert all(document["recordings"][file_id]["load_hp"] == 3 for file_id in document["test_recordings"])
