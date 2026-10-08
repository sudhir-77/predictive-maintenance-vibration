"""Reproduce C-MAPSS remaining-useful-life results and the CWRU vibration layer.

Set thread caps before NumPy and PyTorch are imported.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import json
import sys
import time
from pathlib import Path

from src.hardware import PeakRSS
from src.pipeline import run_pipeline
from src.report import render_resume_notes, update_readme, write_json
from vibration.pipeline import merge_vibration_metrics, run_vibration


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predictive maintenance on NASA C-MAPSS")
    parser.add_argument("--quick", action="store_true", help="Small engine subset for tests. Writes to results/quick.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="CNN device. Default is CPU.")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip the pytest coverage measurement.")
    parser.add_argument("--output", type=Path, default=None, help="Directory for metrics, tables, and plots.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    output_dir = args.output
    if output_dir is None:
        output_dir = root / "results" / "quick" if args.quick else root / "results"
    started = time.perf_counter()
    monitor = PeakRSS()
    monitor.start()
    parent_code = run_pipeline(
        root=root,
        quick=args.quick,
        device=args.device,
        skip_coverage=args.skip_coverage,
        output_dir=output_dir,
    )
    summary = run_vibration(root=root, quick=args.quick, device=args.device, output_dir=output_dir)
    metrics_path = output_dir / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    coverage = payload["metrics"].get("coverage_src_and_vibration_percent")
    merged = merge_vibration_metrics(metrics_path, summary, coverage, quick=args.quick)
    monitor.stop()
    merged["hardware"]["peak_rss_gb"] = round(monitor.gb, 4)
    merged["runtime_seconds"] = round(time.perf_counter() - started, 3)
    written = write_json(metrics_path, merged)
    readme = root / "README.md"
    if readme.exists() and not args.quick:
        update_readme(readme, written)
        (root / "RESUME_NOTES.md").write_text(render_resume_notes(written), encoding="utf-8")
    failed = [item["name"] for item in written["success_criteria"] if not item["passed"]]
    print(f"runtime_seconds={written['runtime_seconds']} peak_rss_gb={written['hardware']['peak_rss_gb']}")
    for item in written["success_criteria"]:
        print(f"  {item['name']}: {item['value']} passed={item['passed']}")
    if args.quick:
        return parent_code
    if failed or parent_code != 0:
        print("criteria not passed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
