"""Metric definitions used by training, tests, and the written artifacts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import (
    ALERT_LEAD_CYCLES,
    ALERT_LEAD_FRACTION,
    COVERAGE_MIN,
    FALSE_ALERT_RUL,
    FALSE_ALERTS_MAX,
    FD001_RMSE_MAX,
    FD003_RMSE_MAX,
)


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: {truth.shape} vs {pred.shape}")
    if truth.size == 0:
        raise ValueError("rmse of an empty array")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("rmse inputs must be finite")
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def compare(value: float | None, op: str, threshold: float, threshold_high: float | None = None) -> bool:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return False
    if op == "<=":
        return float(value) <= threshold
    if op == ">=":
        return float(value) >= threshold
    if op == "<":
        return float(value) < threshold
    if op == ">":
        return float(value) > threshold
    if op == "between":
        if threshold_high is None:
            raise ValueError("between requires threshold_high")
        return threshold <= float(value) <= threshold_high
    raise ValueError(f"unknown op {op}")


def criterion(name: str, value: float | None, op: str, threshold: float, threshold_high: float | None = None) -> dict:
    stored = None if value is None else round(float(value), 6)
    return {
        "name": name,
        "value": stored,
        "op": op,
        "threshold": threshold,
        "threshold_high": threshold_high,
        "passed": compare(stored, op, threshold, threshold_high),
    }


def success_criteria(fd001_rmse: float | None, fd003_rmse: float | None, lead_fraction: float | None, false_alerts: float | None, coverage: float | None) -> list[dict]:
    return [
        criterion("FD001 test RUL RMSE", fd001_rmse, "<=", FD001_RMSE_MAX),
        criterion("FD003 test RUL RMSE", fd003_rmse, "<=", FD003_RMSE_MAX),
        criterion("Alert lead time", lead_fraction, ">=", ALERT_LEAD_FRACTION),
        criterion("False alerts", false_alerts, "<=", FALSE_ALERTS_MAX),
        criterion("Test coverage of src/", coverage, ">=", COVERAGE_MIN),
    ]


def score_alert_arrays(
    alerts: list[np.ndarray],
    ruls: list[np.ndarray],
    lead_min: int = ALERT_LEAD_CYCLES,
    false_rul: float = FALSE_ALERT_RUL,
) -> dict[str, float]:
    """Score alert flags against true remaining life.

    Lead time is the true RUL at the first alert. An engine with no alert has lead time 0.
    A false-alert episode is a contiguous run of alerts that starts while true RUL is still
    above `false_rul` (before the final 50 cycles when false_rul is 50).
    """
    if len(alerts) != len(ruls):
        raise ValueError("alerts and ruls must have one entry per engine")
    n = len(alerts)
    if n == 0:
        return {
            "lead_time_fraction": 0.0,
            "false_alerts_per_engine": 0.0,
            "false_alert_cycles_per_engine": 0.0,
            "median_lead_time": 0.0,
            "mean_lead_time": 0.0,
            "n_engines": 0.0,
        }
    leads: list[float] = []
    false_eps = 0
    false_cycles = 0
    for alert, rul in zip(alerts, ruls):
        flag = np.asarray(alert, dtype=bool)
        remaining = np.asarray(rul, dtype=np.float64)
        if flag.shape != remaining.shape:
            raise ValueError("alert and rul length mismatch")
        if flag.any():
            leads.append(float(remaining[int(np.flatnonzero(flag)[0])]))
        else:
            leads.append(0.0)
        prev = False
        for is_alert, remaining_now in zip(flag, remaining):
            if is_alert and remaining_now > false_rul:
                false_cycles += 1
            if is_alert and not prev and remaining_now > false_rul:
                false_eps += 1
            prev = bool(is_alert)
    lead_arr = np.asarray(leads, dtype=np.float64)
    return {
        "lead_time_fraction": float(np.mean(lead_arr >= lead_min)),
        "false_alerts_per_engine": false_eps / n,
        "false_alert_cycles_per_engine": false_cycles / n,
        "median_lead_time": float(np.median(lead_arr)),
        "mean_lead_time": float(np.mean(lead_arr)),
        "n_engines": float(n),
    }


def summarize_alerts(cycles: pd.DataFrame, lead_min: int = ALERT_LEAD_CYCLES, false_rul: float = FALSE_ALERT_RUL) -> pd.DataFrame:
    """One row per engine, derived from cycle-level alert flags."""
    rows: list[dict] = []
    if cycles.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "unit_id",
                "lead_time",
                "first_alert_cycle",
                "first_alert_rul",
                "false_alert_episodes",
                "false_alert_cycles",
                "meets_lead_time_15",
                "n_cycles",
            ]
        )
    grouped = cycles.groupby(["dataset", "unit_id"], sort=False)
    for (dataset, unit_id), group in grouped:
        ordered = group.sort_values("cycle")
        flag = ordered["alert"].astype(bool).to_numpy()
        remaining = ordered["rul_true"].to_numpy(dtype=np.float64)
        cycle = ordered["cycle"].to_numpy()
        stats = score_alert_arrays([flag], [remaining], lead_min=lead_min, false_rul=false_rul)
        if flag.any():
            index = int(np.flatnonzero(flag)[0])
            first_cycle: int | None = int(cycle[index])
            first_rul: float | None = float(remaining[index])
        else:
            first_cycle = None
            first_rul = None
        rows.append(
            {
                "dataset": dataset,
                "unit_id": int(unit_id),
                "lead_time": stats["median_lead_time"],
                "first_alert_cycle": first_cycle,
                "first_alert_rul": first_rul,
                "false_alert_episodes": int(round(stats["false_alerts_per_engine"])),
                "false_alert_cycles": int(round(stats["false_alert_cycles_per_engine"])),
                "meets_lead_time_15": int(stats["lead_time_fraction"] >= 1.0),
                "n_cycles": int(len(ordered)),
            }
        )
    return pd.DataFrame(rows)


def last_cycle_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["unit_id", "cycle"])
    return ordered.groupby("unit_id", sort=False).tail(1).reset_index(drop=True)
