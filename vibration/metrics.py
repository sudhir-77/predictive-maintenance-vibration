"""Scores recomputed from the prediction table."""

from __future__ import annotations

from sklearn.metrics import f1_score

from src.metrics import criterion
from vibration.config import (
    BASELINE_MARGIN,
    BOTH_HIGH_F1,
    BPFO_REL_ERROR_MAX,
    COVERAGE_MIN,
    MACRO_F1_MIN,
    PARENT_FALSE_ALERTS,
    PARENT_FD001_RMSE,
    PARENT_FD003_RMSE,
    PARENT_LEAD_FRACTION,
    PARENT_REL_TOL,
)


def macro_f1(y_true: list[str] | tuple[str, ...], y_pred: list[str] | tuple[str, ...], labels: list[str] | tuple[str, ...]) -> float:
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("predictions and labels must be the same non-empty length")
    score = f1_score(list(y_true), list(y_pred), average="macro", labels=list(labels), zero_division=0)
    return float(score)


def relative_deviation(value: float, reference: float) -> float:
    if abs(reference) < 1e-12:
        return abs(float(value) - float(reference))
    return abs(float(value) - float(reference)) / abs(float(reference))


def parent_headline_deviation(fd001_rmse: float, fd003_rmse: float, lead_fraction: float, false_alerts: float) -> float:
    return max(
        relative_deviation(fd001_rmse, PARENT_FD001_RMSE),
        relative_deviation(fd003_rmse, PARENT_FD003_RMSE),
        relative_deviation(lead_fraction, PARENT_LEAD_FRACTION),
        relative_deviation(false_alerts, PARENT_FALSE_ALERTS),
    )


def baseline_comparison_passed(best_f1: float, baseline_f1: float) -> bool:
    margin = best_f1 + 1e-12 >= baseline_f1 + BASELINE_MARGIN
    both_high = best_f1 + 1e-12 >= BOTH_HIGH_F1 and baseline_f1 + 1e-12 >= BOTH_HIGH_F1
    return bool(margin or both_high)


def vibration_criteria(
    best_f1: float | None,
    baseline_f1: float | None,
    rel_error: float | None,
    overlap_count: int | None,
    parent_deviation: float | None,
    parent_criteria_passed: bool,
    coverage: float | None,
    n_classes: int,
) -> list[dict]:
    if best_f1 is None or baseline_f1 is None:
        comparison = criterion("Best model vs. feature baseline", None, ">=", BASELINE_MARGIN)
    else:
        use_high_bar = best_f1 >= BOTH_HIGH_F1 and baseline_f1 >= BOTH_HIGH_F1 and best_f1 < baseline_f1 + BASELINE_MARGIN
        threshold = BOTH_HIGH_F1 if use_high_bar else baseline_f1 + BASELINE_MARGIN
        comparison = criterion("Best model vs. feature baseline", best_f1, ">=", threshold)
        comparison["passed"] = baseline_comparison_passed(best_f1, baseline_f1)
    f1_name = f"Macro-F1 on held-out motor load ({n_classes} classes)"
    parent = criterion("Parent project", parent_deviation, "<=", PARENT_REL_TOL)
    parent["passed"] = bool(parent["passed"] and parent_criteria_passed)
    return [
        criterion(f1_name, best_f1, ">=", MACRO_F1_MIN),
        comparison,
        criterion("Envelope fault-frequency check", rel_error, "<=", BPFO_REL_ERROR_MAX),
        criterion("No leakage", None if overlap_count is None else float(overlap_count), "<=", 0.0),
        parent,
        criterion("Test coverage of src/ and vibration/", coverage, ">=", COVERAGE_MIN),
    ]
