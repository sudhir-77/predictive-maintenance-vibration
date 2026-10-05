"""End-to-end run: train, write artifacts, then measure test coverage."""

from __future__ import annotations

import gc
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.artifacts import assemble_metrics, test_prediction_table, val_cycle_table
from src.config import DATASETS, PROJECT_SLUG
from src.dashboard import write_dashboard
from src.download import ensure_cmapss
from src.hardware import PeakRSS, configure_torch_threads, hardware_dict
from src.plots import write_plots
from src.report import render_resume_notes, update_readme, write_json
from src.train_eval import evaluate_dataset


def measure_coverage(root: Path) -> tuple[float | None, int, str]:
    env = os.environ.copy()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=src",
        "--cov-report=term-missing",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
    ]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, env=env)
    text = completed.stdout + "\n" + completed.stderr
    match = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+(?:\.\d+)?)%", text, re.MULTILINE)
    percent = float(match.group(1)) if match else None
    return percent, int(completed.returncode), text


def run_pipeline(
    root: Path,
    quick: bool = False,
    device: str = "cpu",
    skip_coverage: bool = False,
    output_dir: Path | None = None,
    score_test: bool = True,
) -> int:
    started = time.perf_counter()
    configure_torch_threads()
    monitor = PeakRSS()
    monitor.start()
    if output_dir is None:
        output_dir = root / "results" / "quick" if quick else root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ensure_cmapss(root)
    splits_dir = root / "data" / "splits"
    results = [
        evaluate_dataset(data_dir, splits_dir, dataset, quick, device, score_test)
        for dataset in DATASETS
    ]
    by_name = {result.info["dataset"]: result for result in results}
    fd001_test = test_prediction_table(by_name["FD001"].test_predictions) if len(by_name["FD001"].test_predictions) else pd.DataFrame()
    fd003_test = test_prediction_table(by_name["FD003"].test_predictions) if len(by_name["FD003"].test_predictions) else pd.DataFrame()
    cycles = val_cycle_table(pd.concat([by_name["FD001"].val_cycles, by_name["FD003"].val_cycles], ignore_index=True))
    infos = [by_name[name].info for name in DATASETS]
    baselines = [by_name[name].baseline for name in DATASETS]
    del results, by_name
    gc.collect()

    if len(fd001_test):
        fd001_test.to_csv(output_dir / "fd001_test_predictions.csv", index=False)
    if len(fd003_test):
        fd003_test.to_csv(output_dir / "fd003_test_predictions.csv", index=False)
    cycles.to_csv(output_dir / "val_alert_cycles.csv", index=False)
    # Recompute from the files that were just written.
    if len(fd001_test):
        fd001_test = pd.read_csv(output_dir / "fd001_test_predictions.csv")
    if len(fd003_test):
        fd003_test = pd.read_csv(output_dir / "fd003_test_predictions.csv")
    cycles = pd.read_csv(output_dir / "val_alert_cycles.csv")

    hardware = hardware_dict()
    payload, engines = assemble_metrics(
        fd001_test,
        fd003_test,
        cycles,
        infos,
        baselines,
        hardware,
        runtime_seconds=0.0,
        coverage=None,
        quick=quick,
    )
    engines.to_csv(output_dir / "val_engine_alerts.csv", index=False)
    write_plots(output_dir, fd001_test, fd003_test, cycles, engines, payload)

    coverage = None
    pytest_code = 0
    if not skip_coverage:
        print("measuring test coverage")
        coverage, pytest_code, log_text = measure_coverage(root)
        (output_dir / "pytest_coverage.log").write_text(log_text, encoding="utf-8")
        print(log_text[-2000:])
    payload, engines = assemble_metrics(
        fd001_test,
        fd003_test,
        cycles,
        infos,
        baselines,
        hardware,
        runtime_seconds=0.0,
        coverage=coverage,
        quick=quick,
    )
    if pytest_code != 0:
        for item in payload["success_criteria"]:
            if item["name"] == "Test coverage of src/":
                item["passed"] = False
        payload["metrics"]["pytest_exit_code"] = pytest_code
    monitor.stop()
    hardware["peak_rss_gb"] = round(monitor.gb, 4)
    payload["hardware"] = hardware
    payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
    payload["project"] = PROJECT_SLUG
    written = write_json(output_dir / "metrics.json", payload)
    write_dashboard(output_dir / "dashboard.html", written, fd001_test, fd003_test, cycles, engines)
    readme = root / "README.md"
    if readme.exists() and not quick:
        update_readme(readme, written)
        notes = render_resume_notes(written)
        (root / "RESUME_NOTES.md").write_text(notes, encoding="utf-8")
    failed = [item["name"] for item in written["success_criteria"] if not item["passed"]]
    print(f"runtime_seconds={written['runtime_seconds']} peak_rss_gb={hardware['peak_rss_gb']}")
    for item in written["success_criteria"]:
        print(f"  {item['name']}: {item['value']} passed={item['passed']}")
    if failed:
        print("criteria not passed: " + ", ".join(failed))
        return 1
    return 0
