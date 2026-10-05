"""CLI smoke test on a small subset. Does not require the official thresholds."""

from __future__ import annotations

import json
import time

from src.pipeline import run_pipeline


def test_quick_pipeline_writes_recomputable_artifacts(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    run_pipeline(
        root=root,
        quick=True,
        device="cpu",
        skip_coverage=True,
        output_dir=tmp_path,
        score_test=True,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 180
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["project"] == "predictive-maintenance-ops"
    assert {item["name"] for item in metrics["success_criteria"]} == {
        "FD001 test RUL RMSE",
        "FD003 test RUL RMSE",
        "Alert lead time",
        "False alerts",
        "Test coverage of src/",
    }
    for name in ("fd001_test_predictions.csv", "fd003_test_predictions.csv", "val_alert_cycles.csv", "val_engine_alerts.csv", "dashboard.html"):
        assert (tmp_path / name).exists()
    fd001 = (tmp_path / "fd001_test_predictions.csv").read_text(encoding="utf-8").splitlines()
    assert fd001[0].split(",")[0] == "row_id"
    assert "rul_true" in fd001[0] and "rul_pred" in fd001[0]
    assert len(fd001) == 5  # header + 4 quick test engines
    split = json.loads((root / "data" / "splits" / "FD001-quick.json").read_text(encoding="utf-8"))
    assert set(split["train_units"]).isdisjoint(split["val_units"])
    assert metrics["hardware"]["peak_rss_gb"] >= 0
