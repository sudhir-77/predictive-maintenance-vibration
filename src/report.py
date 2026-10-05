"""README results block and resume notes, both filled from metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

BANNED = ("leveraged", "robust", "cutting-edge", "seamless", "spearheaded", "\u2014")


def to_builtin(value):
    import math

    import numpy as np

    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def render_results_markdown(metrics: dict) -> str:
    lines = [
        "| Criterion | Value | Op | Threshold | Passed |",
        "|---|---:|---|---:|---|",
    ]
    failed = []
    for item in metrics["success_criteria"]:
        value = "n/a" if item["value"] is None else item["value"]
        passed = "yes" if item["passed"] else "no"
        if not item["passed"]:
            failed.append(item["name"])
        lines.append(f"| {item['name']} | {value} | {item['op']} | {item['threshold']} | {passed} |")
    alerts = metrics["metrics"]["alerts"]
    lines.append("")
    lines.append(
        f"Validation engines: median lead time {alerts['median_lead_time']} cycles, "
        f"mean lead time {alerts['mean_lead_time']} cycles, "
        f"{alerts['false_alerts_per_engine']} false alert episodes per engine."
    )
    fd001 = metrics["metrics"]["fd001"]
    fd003 = metrics["metrics"]["fd003"]
    if fd001.get("test_rmse") is not None:
        lines.append("")
        lines.append("| Dataset | Selected RMSE | LightGBM | CNN | Constant baseline | Piecewise baseline |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, block in (("FD001", fd001), ("FD003", fd003)):
            lines.append(
                f"| {name} | {block.get('test_rmse')} | {block.get('test_rmse_lgbm')} | "
                f"{block.get('test_rmse_cnn')} | {block.get('test_rmse_baseline_constant')} | "
                f"{block.get('test_rmse_baseline_piecewise')} |"
            )
    lines.append("")
    lines.append(f"Runtime seconds: {metrics['runtime_seconds']}.")
    peak = metrics["hardware"].get("peak_rss_gb")
    if peak is not None:
        lines.append(f"Peak RSS GB: {peak}.")
    if failed:
        lines.append("")
        lines.append("Criteria that did not pass: " + ", ".join(failed) + ".")
    else:
        lines.append("")
        lines.append("Every success criterion passed.")
    return "\n".join(lines) + "\n"


def update_readme(path: Path, metrics: dict) -> None:
    text = path.read_text(encoding="utf-8")
    start = "<!-- RESULTS_START -->"
    end = "<!-- RESULTS_END -->"
    block = start + "\n" + render_results_markdown(metrics) + end
    if start not in text or end not in text:
        text = text.rstrip() + "\n\n" + block + "\n"
    else:
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        text = pre + block + post
    path.write_text(text, encoding="utf-8")


def render_resume_notes(metrics: dict) -> str:
    headline = metrics["metrics"]["headline_text"]
    fd001 = headline["fd001_test_rmse"]
    fd003 = headline["fd003_test_rmse"]
    baseline = headline["fd003_baseline_constant_rmse"]
    lead = headline["alert_median_lead_cycles"]
    percent = headline["alert_engines_percent"]
    false_alerts = headline["false_alerts_per_engine"]
    text = f"""# Resume notes

## Bullets

- Predicted remaining useful life on NASA C-MAPSS engine telemetry with LightGBM and a one-dimensional CNN, reaching {fd001} RMSE (cycles) on the official FD001 test set.
- Designed an alert policy combining RUL forecasts with control-chart and isolation-forest anomaly scores that warned {lead} cycles ahead for {percent}% of engines at {false_alerts} false alerts per engine.
- On FD003, the same models reached {fd003} test RMSE (cycles), compared with {baseline} for a constant mean-RUL baseline.
- Built the SQL feature layer in DuckDB and a static operations dashboard for alert review.

## Skills and tools

Python, SQL, DuckDB, LightGBM, PyTorch, scikit-learn, pandas, statistics, anomaly detection, control charts, time series, predictive maintenance, matplotlib, Plotly.

## Summary

This project predicts how many cycles a turbofan engine has left using NASA C-MAPSS sensor streams. It builds rolling features in SQL, compares gradient boosting with a one-dimensional convolutional network, and checks an alert rule against the actual failure times.

## GitHub description

Predict remaining useful life and failure alerts from NASA C-MAPSS turbofan telemetry.
"""
    lowered = text.lower()
    for word in BANNED:
        if word.lower() in lowered:
            raise ValueError(f"resume notes contain banned wording: {word}")
    return text


def write_json(path: Path, payload: dict) -> dict:
    clean = to_builtin(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))
