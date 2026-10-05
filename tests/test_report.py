"""Resume notes and the generated results table."""

from __future__ import annotations

import json

from src.report import render_resume_notes, render_results_markdown, update_readme


def _metrics() -> dict:
    return {
        "success_criteria": [
            {"name": "FD001 test RUL RMSE", "value": 12.3456, "op": "<=", "threshold": 18.0, "passed": True},
            {"name": "False alerts", "value": 0.05, "op": "<=", "threshold": 0.1, "passed": True},
        ],
        "metrics": {
            "datasets": ["FD001", "FD003"],
            "fd001": {"test_rmse": 12.34, "test_rmse_lgbm": 12.5, "test_rmse_cnn": 13.0, "test_rmse_baseline_constant": 40.0, "test_rmse_baseline_piecewise": 30.0},
            "fd003": {"test_rmse": 16.5, "test_rmse_lgbm": 16.7, "test_rmse_cnn": 17.2, "test_rmse_baseline_constant": 41.0, "test_rmse_baseline_piecewise": 31.0},
            "alerts": {
                "median_lead_time": 28.5,
                "mean_lead_time": 30.0,
                "false_alerts_per_engine": 0.05,
            },
            "headline_text": {
                "fd001_test_rmse": "12.34",
                "fd003_test_rmse": "16.50",
                "fd001_baseline_constant_rmse": "40.12",
                "fd003_baseline_constant_rmse": "41.00",
                "alert_median_lead_cycles": "28.50",
                "alert_engines_percent": "90.00",
                "false_alerts_per_engine": "0.0500",
            },
        },
        "hardware": {"peak_rss_gb": 0.5},
        "runtime_seconds": 12.5,
    }


def test_resume_numbers_appear_in_metrics_json():
    metrics = _metrics()
    text = render_resume_notes(metrics)
    raw = json.dumps(metrics)
    for token in ("12.34", "16.50", "41.00", "28.50", "90.00", "0.0500", "FD001", "FD003"):
        assert token in text
        assert token in raw
    assert "\u2014" not in text


def test_results_markdown_marks_failures():
    metrics = _metrics()
    metrics["success_criteria"][0]["passed"] = False
    text = render_results_markdown(metrics)
    assert "FD001 test RUL RMSE" in text
    assert "did not pass" in text


def test_readme_marker_is_replaced(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("intro\n<!-- RESULTS_START -->\nPLACEHOLDER_OLD_RESULTS\n<!-- RESULTS_END -->\nend\n", encoding="utf-8")
    update_readme(path, _metrics())
    text = path.read_text(encoding="utf-8")
    assert "PLACEHOLDER_OLD_RESULTS" not in text
    assert "intro" in text
    assert "12.3456" in text
    assert text.strip().endswith("end")
