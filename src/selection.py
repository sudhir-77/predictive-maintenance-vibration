"""Validation-only blend weights and calibrators. Never fit these on the test set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.config import SEED
from src.metrics import rmse


def fit_calibrator(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return intercept and slope for y ≈ a + b * pred. Identity when the fit is unstable."""
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if len(truth) < 5 or float(np.std(pred)) < 1e-6:
        return 0.0, 1.0
    slope, intercept = np.polyfit(pred, truth, 1)
    if not np.isfinite(intercept) or not np.isfinite(slope):
        return 0.0, 1.0
    if slope < 0.25 or slope > 2.5:
        return 0.0, 1.0
    return float(intercept), float(slope)


def apply_calibrator(y_pred: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    pred = np.asarray(y_pred, dtype=np.float64)
    return np.clip(intercept + slope * pred, 0.0, 300.0)


def maybe_calibrate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
    """Keep the calibrator only when it lowers validation RMSE by more than 0.05 cycles."""
    raw = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 300.0)
    intercept, slope = fit_calibrator(y_true, y_pred)
    calibrated = apply_calibrator(y_pred, intercept, slope)
    if rmse(y_true, calibrated) + 0.05 < rmse(y_true, raw):
        return (intercept, slope), calibrated
    return (0.0, 1.0), raw


@dataclass
class Calibrator:
    kind: str
    intercept: float
    slope: float
    isotonic: IsotonicRegression | None = None

    def as_dict(self) -> dict:
        return {"kind": self.kind, "intercept": self.intercept, "slope": self.slope}


def apply_chosen(y_pred: np.ndarray, calibrator: Calibrator) -> np.ndarray:
    pred = np.asarray(y_pred, dtype=np.float64)
    if calibrator.kind == "isotonic":
        if calibrator.isotonic is None:
            raise RuntimeError("isotonic calibrator is missing its model")
        return np.clip(calibrator.isotonic.predict(pred), 0.0, 300.0)
    return apply_calibrator(pred, calibrator.intercept, calibrator.slope)


def _fit_predict(kind: str, x_fit: np.ndarray, y_fit: np.ndarray, x_score: np.ndarray):
    if kind == "isotonic":
        model = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=300.0)
        model.fit(x_fit, y_fit)
        return np.clip(model.predict(x_score), 0.0, 300.0), model
    intercept, slope = fit_calibrator(y_fit, x_fit)
    return apply_calibrator(x_score, intercept, slope), (intercept, slope)


def choose_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> tuple[Calibrator, np.ndarray]:
    """Pick identity, a line, or isotonic regression using validation engines only.

    Isotonic is kept only when a two-fold split by engine beats the line by more
    than 0.3 cycles. The winning map is then refit on every validation row passed in.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    (intercept, slope), linear_hat = maybe_calibrate(truth, pred)
    kind = "identity" if intercept == 0.0 and slope == 1.0 else "linear"
    linear = Calibrator(kind, intercept, slope, None)
    labels = np.asarray(groups)
    unique = np.unique(labels)
    if len(unique) < 4 or len(truth) < 50:
        return linear, linear_hat
    order = np.random.default_rng(SEED).permutation(unique)
    left = np.isin(labels, order[: len(order) // 2])
    folds = ((left, ~left), (~left, left))
    if any(int(fit.sum()) < 20 or int(score.sum()) < 20 for fit, score in folds):
        return linear, linear_hat
    heldout = {"isotonic": [], "linear": []}
    for fit_mask, score_mask in folds:
        for name in ("isotonic", "linear"):
            hat, _model = _fit_predict(name, pred[fit_mask], truth[fit_mask], pred[score_mask])
            heldout[name].append(rmse(truth[score_mask], hat))
    if float(np.mean(heldout["isotonic"])) + 0.3 < float(np.mean(heldout["linear"])):
        hat, model = _fit_predict("isotonic", pred, truth, pred)
        return Calibrator("isotonic", 0.0, 1.0, model), hat
    return linear, linear_hat


def choose_alpha(y_true: np.ndarray, pred_lgbm: np.ndarray, pred_cnn: np.ndarray) -> tuple[float, float]:
    """LightGBM weight on a small grid. Chosen by validation RMSE only."""
    best_alpha = 1.0
    best_score = float("inf")
    left = np.asarray(pred_lgbm, dtype=np.float64)
    right = np.asarray(pred_cnn, dtype=np.float64)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = alpha * left + (1.0 - alpha) * right
        score = rmse(y_true, blended)
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha, best_score
