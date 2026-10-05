"""Reproduce C-MAPSS remaining-useful-life and alert results.

Set thread caps before NumPy and PyTorch are imported.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import sys
from pathlib import Path

from src.pipeline import run_pipeline


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
    return run_pipeline(
        root=root,
        quick=args.quick,
        device=args.device,
        skip_coverage=args.skip_coverage,
        output_dir=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
