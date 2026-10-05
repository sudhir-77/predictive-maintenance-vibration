"""Alert rules tuned only on validation engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ALERT_LEAD_CYCLES, ALERT_LEAD_FRACTION, FALSE_ALERT_RUL, FALSE_ALERTS_MAX
from src.metrics import score_alert_arrays


@dataclass(frozen=True)
class AlertPolicy:
    rul_threshold: float
    persist_cycles: int
    smooth_window: int
    z_threshold: float
    iforest_threshold: float
    anomaly_margin: float
    anomaly_persist: int

    def uses_anomaly(self) -> bool:
        return self.z_threshold < 1e8 or self.iforest_threshold < 1e8

    def as_dict(self) -> dict:
        return {
            "rul_threshold": self.rul_threshold,
            "persist_cycles": self.persist_cycles,
            "smooth_window": self.smooth_window,
            "z_threshold": self.z_threshold,
            "iforest_threshold": self.iforest_threshold,
            "anomaly_margin": self.anomaly_margin,
            "anomaly_persist": self.anomaly_persist,
            "uses_anomaly": self.uses_anomaly(),
        }


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return series.copy()
    return pd.Series(series).rolling(window, min_periods=1).median().to_numpy(dtype=np.float64)


def persist_true(flag: np.ndarray, periods: int) -> np.ndarray:
    """True once `periods` consecutive flags have been true, including the current row."""
    mask = np.asarray(flag, dtype=bool)
    out = np.zeros(len(mask), dtype=bool)
    if periods <= 1:
        return mask.copy()
    if len(mask) < periods:
        return out
    cumulative = np.cumsum(mask.astype(np.int32))
    out[periods - 1] = cumulative[periods - 1] >= periods
    if len(mask) > periods:
        window_sum = cumulative[periods:] - cumulative[:-periods]
        out[periods:] = window_sum >= periods
    return out


def unit_alerts(pred: np.ndarray, control_z: np.ndarray, iforest: np.ndarray, policy: AlertPolicy) -> np.ndarray:
    """Alert when a low RUL forecast persists, or a persisted anomaly agrees that RUL is getting low."""
    smooth = rolling_median(pred, policy.smooth_window)
    low = persist_true(smooth < policy.rul_threshold, policy.persist_cycles)
    anomaly = (np.asarray(control_z, dtype=np.float64) >= policy.z_threshold) | (
        np.asarray(iforest, dtype=np.float64) >= policy.iforest_threshold
    )
    anomaly_persist = persist_true(anomaly, policy.anomaly_persist)
    gated = anomaly_persist & (smooth < (policy.rul_threshold + policy.anomaly_margin))
    return low | gated


def default_policies(iforest_thresholds: dict[str, float], quick: bool) -> list[AlertPolicy]:
    off = 1e12
    if quick:
        rul_thresholds = [30.0, 40.0]
        persists = [3, 6]
        smooths = [5]
        z_thresholds = [8.0, off]
        if_thresholds = [off]
        margins = [15.0]
        anomaly_persists = [4]
    else:
        rul_thresholds = [18.0, 24.0, 30.0, 36.0, 42.0, 50.0]
        persists = [2, 4, 6, 10]
        smooths = [5, 11]
        z_thresholds = [6.0, 10.0, off]
        if_thresholds = [float(iforest_thresholds["q99"]), float(iforest_thresholds["q999"]), off]
        margins = [12.0, 24.0]
        anomaly_persists = [4]
    policies: list[AlertPolicy] = []
    seen: set[tuple] = set()
    for rul_threshold in rul_thresholds:
        for persist in persists:
            for smooth in smooths:
                for z_threshold in z_thresholds:
                    for if_threshold in if_thresholds:
                        for margin in margins:
                            for anomaly_persist in anomaly_persists:
                                # Disabled anomaly does not depend on margin or anomaly persistence.
                                if z_threshold >= 1e8 and if_threshold >= 1e8:
                                    key = (rul_threshold, persist, smooth, off, off, 0.0, 1)
                                    policy = AlertPolicy(rul_threshold, persist, smooth, off, off, 0.0, 1)
                                else:
                                    key = (rul_threshold, persist, smooth, z_threshold, if_threshold, margin, anomaly_persist)
                                    policy = AlertPolicy(
                                        rul_threshold,
                                        persist,
                                        smooth,
                                        z_threshold,
                                        if_threshold,
                                        margin,
                                        anomaly_persist,
                                    )
                                if key not in seen:
                                    seen.add(key)
                                    policies.append(policy)
    return policies


def _split_engines(frame: pd.DataFrame) -> list[dict]:
    engines: list[dict] = []
    for _, group in frame.groupby("engine_key", sort=False):
        ordered = group.sort_values("cycle")
        engines.append(
            {
                "index": ordered.index.to_numpy(),
                "rul": ordered["rul_true"].to_numpy(dtype=np.float64),
                "pred": ordered["rul_pred"].to_numpy(dtype=np.float64),
                "z": ordered["control_chart_z"].to_numpy(dtype=np.float64),
                "ifo": ordered["iforest_score"].to_numpy(dtype=np.float64),
            }
        )
    return engines


def apply_policy(frame: pd.DataFrame, policy: AlertPolicy) -> pd.Series:
    alerts = pd.Series(False, index=frame.index)
    for engine in _split_engines(frame):
        flag = unit_alerts(engine["pred"], engine["z"], engine["ifo"], policy)
        alerts.loc[engine["index"]] = flag
    return alerts


def tune_policy(
    frame: pd.DataFrame,
    iforest_thresholds: dict[str, float],
    quick: bool = False,
    policies: list[AlertPolicy] | None = None,
) -> tuple[AlertPolicy, dict]:
    """Pick a policy on validation engines. Test rows must not be in `frame`."""
    if (frame["origin"] == "test_file").any() or (frame["split"] == "test").any():
        raise RuntimeError("alert thresholds cannot be tuned on test engines")
    engines = _split_engines(frame)
    if not engines:
        raise ValueError("no validation engines to tune")
    # Smooth once per window so the grid does not recompute medians.
    smooth_windows = sorted({policy.smooth_window for policy in (policies or default_policies(iforest_thresholds, quick))})
    for engine in engines:
        engine["smooth"] = {window: rolling_median(engine["pred"], window) for window in smooth_windows}
    chosen = policies if policies is not None else default_policies(iforest_thresholds, quick)
    best: tuple | None = None
    for policy in chosen:
        flags = []
        for engine in engines:
            smooth = engine["smooth"][policy.smooth_window]
            low = persist_true(smooth < policy.rul_threshold, policy.persist_cycles)
            anomaly = (engine["z"] >= policy.z_threshold) | (engine["ifo"] >= policy.iforest_threshold)
            gated = persist_true(anomaly, policy.anomaly_persist) & (smooth < (policy.rul_threshold + policy.anomaly_margin))
            flags.append(low | gated)
        stats = score_alert_arrays(
            flags,
            [engine["rul"] for engine in engines],
            lead_min=ALERT_LEAD_CYCLES,
            false_rul=FALSE_ALERT_RUL,
        )
        passes = (
            stats["lead_time_fraction"] >= ALERT_LEAD_FRACTION
            and stats["false_alerts_per_engine"] <= FALSE_ALERTS_MAX
            and stats["false_alert_cycles_per_engine"] <= FALSE_ALERTS_MAX
        )
        penalty = (
            100.0 * max(0.0, ALERT_LEAD_FRACTION - stats["lead_time_fraction"])
            + 100.0 * max(0.0, stats["false_alerts_per_engine"] - FALSE_ALERTS_MAX)
            + 20.0 * max(0.0, stats["false_alert_cycles_per_engine"] - FALSE_ALERTS_MAX)
        )
        # Passing policies come first. Among them, prefer a rule that actually uses the anomaly signal,
        # then higher lead-time coverage, then fewer false alerts, then a longer median lead.
        rank = (
            0 if passes else 1,
            0 if (passes and policy.uses_anomaly()) else 1,
            penalty,
            -stats["lead_time_fraction"],
            stats["false_alerts_per_engine"],
            -stats["median_lead_time"],
        )
        if best is None or rank < best[0]:
            best = (rank, policy, stats, passes)
    assert best is not None
    policy = best[1]
    stats = dict(best[2])
    stats["passed_constraints"] = bool(best[3])
    return policy, stats
