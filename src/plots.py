"""PNG figures written next to metrics.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pred_vs_actual(test: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(test["rul_true"], test["rul_pred"], s=18, alpha=0.8, c="#1f4e79")
    limit = float(max(test["rul_true"].max(), test["rul_pred"].max(), 1))
    ax.plot([0, limit], [0, limit], color="#b85c38", linewidth=1)
    ax.set_xlabel("Official RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(title)
    _save(fig, path)


def plot_rmse_bars(metrics: dict, path: Path) -> None:
    labels = ["Selected", "LightGBM", "CNN", "Constant", "Piecewise"]
    keys = [
        "test_rmse",
        "test_rmse_lgbm",
        "test_rmse_cnn",
        "test_rmse_baseline_constant",
        "test_rmse_baseline_piecewise",
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    import numpy as np

    x = np.arange(len(labels))
    width = 0.36
    for offset, name, color in ((-width / 2, "fd001", "#1f4e79"), (width / 2, "fd003", "#5b8c5a")):
        block = metrics["metrics"][name]
        values = [0.0 if block.get(key) is None else block.get(key) for key in keys]
        ax.bar(x + offset, values, width=width, label=name.upper(), color=color)
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Test RMSE (cycles)")
    ax.set_title("RUL error on the official test engines")
    ax.legend()
    _save(fig, path)


def plot_lead_times(engines: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for dataset, color in (("FD001", "#1f4e79"), ("FD003", "#5b8c5a")):
        subset = engines[engines["dataset"] == dataset]
        ax.hist(subset["lead_time"], bins=12, alpha=0.65, label=dataset, color=color)
    ax.axvline(15, color="#b85c38", linestyle="--", label="15-cycle mark")
    ax.set_xlabel("Lead time (cycles before failure)")
    ax.set_ylabel("Validation engines")
    ax.set_title("Alert lead time")
    ax.legend()
    _save(fig, path)


def plot_example_engine(cycles: pd.DataFrame, engines: pd.DataFrame, path: Path) -> None:
    if engines.empty:
        return
    pool = engines[engines["dataset"] == "FD001"]
    if pool.empty:
        pool = engines
    median = float(pool["lead_time"].median())
    pick = pool.iloc[(pool["lead_time"] - median).abs().argmin()]
    subset = cycles[(cycles["dataset"] == pick["dataset"]) & (cycles["unit_id"] == pick["unit_id"])].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(subset["cycle"], subset["rul_true"], label="True RUL", color="#333333")
    ax.plot(subset["cycle"], subset["rul_pred"], label="Predicted RUL", color="#1f4e79")
    alerts = subset[subset["alert"].astype(bool)]
    if len(alerts):
        ax.scatter(alerts["cycle"], alerts["rul_pred"], s=12, color="#b85c38", label="Alert", zorder=3)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Example validation engine {pick['dataset']} unit {int(pick['unit_id'])}")
    ax.legend()
    _save(fig, path)


def write_plots(output_dir: Path, fd001: pd.DataFrame, fd003: pd.DataFrame, cycles: pd.DataFrame, engines: pd.DataFrame, metrics: dict) -> None:
    if len(fd001):
        plot_pred_vs_actual(fd001, output_dir / "fd001_pred_vs_actual.png", "FD001 official test")
    if len(fd003):
        plot_pred_vs_actual(fd003, output_dir / "fd003_pred_vs_actual.png", "FD003 official test")
    if len(fd001) and len(fd003):
        plot_rmse_bars(metrics, output_dir / "rmse_comparison.png")
    if len(engines):
        plot_lead_times(engines, output_dir / "alert_lead_times.png")
        plot_example_engine(cycles, engines, output_dir / "example_engine.png")
