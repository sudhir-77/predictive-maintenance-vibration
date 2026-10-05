"""Tables an independent evaluator can recompute metrics from."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import rmse, score_alert_arrays, success_criteria, summarize_alerts


def make_row_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["dataset"].astype(str)
        + "-"
        + frame["origin"].astype(str)
        + "-u"
        + frame["unit_id"].astype(int).astype(str)
        + "-c"
        + frame["cycle"].astype(int).astype(str)
        + "-r"
        + frame["source_row"].astype(int).astype(str)
    )


def test_prediction_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "row_id",
        "source_row",
        "dataset",
        "unit_id",
        "cycle",
        "rul_true",
        "rul_pred",
        "rul_pred_lgbm",
        "rul_pred_cnn",
        "rul_pred_baseline_constant",
        "rul_pred_baseline_piecewise",
    ]
    out = frame.copy()
    out["row_id"] = make_row_id(out)
    return out[columns].sort_values(["dataset", "unit_id"]).reset_index(drop=True)


def val_cycle_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "row_id",
        "source_row",
        "dataset",
        "unit_id",
        "cycle",
        "rul_true",
        "rul_pred",
        "rul_pred_lgbm",
        "rul_pred_cnn",
        "control_chart_z",
        "iforest_score",
        "anomaly_flag",
        "alert",
    ]
    out = frame.copy()
    out["row_id"] = make_row_id(out)
    out["alert"] = out["alert"].astype(int)
    out["anomaly_flag"] = out["anomaly_flag"].astype(int)
    return out[columns].sort_values(["dataset", "unit_id", "cycle"]).reset_index(drop=True)


def _per_dataset_alert(cycles: pd.DataFrame, dataset: str) -> dict[str, float]:
    subset = cycles[cycles["dataset"] == dataset]
    flags: list[np.ndarray] = []
    ruls: list[np.ndarray] = []
    for _, group in subset.groupby("unit_id", sort=False):
        ordered = group.sort_values("cycle")
        flags.append(ordered["alert"].astype(bool).to_numpy())
        ruls.append(ordered["rul_true"].to_numpy(dtype=np.float64))
    return score_alert_arrays(flags, ruls)


def pooled_alert_stats(cycles: pd.DataFrame) -> dict[str, float]:
    flags: list[np.ndarray] = []
    ruls: list[np.ndarray] = []
    for _, group in cycles.groupby(["dataset", "unit_id"], sort=False):
        ordered = group.sort_values("cycle")
        flags.append(ordered["alert"].astype(bool).to_numpy())
        ruls.append(ordered["rul_true"].to_numpy(dtype=np.float64))
    return score_alert_arrays(flags, ruls)


def _fmt(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def assemble_metrics(
    fd001_test: pd.DataFrame,
    fd003_test: pd.DataFrame,
    cycles: pd.DataFrame,
    infos: list[dict],
    baselines: list[dict],
    hardware: dict,
    runtime_seconds: float,
    coverage: float | None,
    quick: bool,
) -> tuple[dict, pd.DataFrame]:
    """Compute every reported number from the prediction tables."""
    engines = summarize_alerts(cycles)
    alert_stats = pooled_alert_stats(cycles)
    by_name = {info["dataset"]: info for info in infos}
    base_by_name = {item["dataset"]: item for item in baselines}

    def pack(dataset: str, test: pd.DataFrame) -> dict:
        info = by_name[dataset]
        base = base_by_name[dataset]
        packed = {
            "val_truncated_rmse": info["val_truncated_rmse"],
            "blend_alpha_lgbm": info["blend_alpha_lgbm"],
            "lgbm_iterations": info["lgbm_iterations"],
            "cnn_epochs": info["cnn_epochs"],
            "n_features": info["n_features"],
            "calibrator_lgbm_intercept": info["calibrator_lgbm"]["intercept"],
            "calibrator_lgbm_slope": info["calibrator_lgbm"]["slope"],
            "calibrator_cnn_intercept": info["calibrator_cnn"]["intercept"],
            "calibrator_cnn_slope": info["calibrator_cnn"]["slope"],
            "alert_policy": info["alert_policy"],
            "alert_tune_passed": info["alert_tune"]["passed_constraints"],
            "constant_mean_rul": base["constant_mean_rul"],
            "piecewise_sensor": base["piecewise_sensor"],
            "baseline_val_lead_time_fraction": base["val_alert_lead_time_fraction"],
            "baseline_val_false_alerts_per_engine": base["val_false_alerts_per_engine"],
        }
        if test is not None and len(test) > 0:
            packed["test_rmse"] = rmse(test["rul_true"], test["rul_pred"])
            packed["test_rmse_lgbm"] = rmse(test["rul_true"], test["rul_pred_lgbm"])
            packed["test_rmse_cnn"] = rmse(test["rul_true"], test["rul_pred_cnn"])
            packed["test_rmse_baseline_constant"] = rmse(test["rul_true"], test["rul_pred_baseline_constant"])
            packed["test_rmse_baseline_piecewise"] = rmse(test["rul_true"], test["rul_pred_baseline_piecewise"])
            packed["beats_constant_baseline"] = packed["test_rmse"] < packed["test_rmse_baseline_constant"]
            packed["n_test_engines"] = int(len(test))
        else:
            packed["test_rmse"] = None
            packed["n_test_engines"] = 0
        per = _per_dataset_alert(cycles, dataset)
        packed["val_lead_time_fraction"] = per["lead_time_fraction"]
        packed["val_false_alerts_per_engine"] = per["false_alerts_per_engine"]
        packed["val_median_lead_time"] = per["median_lead_time"]
        return packed

    fd001 = pack("FD001", fd001_test)
    fd003 = pack("FD003", fd003_test)
    lead_fraction = alert_stats["lead_time_fraction"]
    false_alerts = alert_stats["false_alerts_per_engine"]
    headline = {
        "fd001_test_rmse": None if fd001["test_rmse"] is None else _fmt(fd001["test_rmse"], 2),
        "fd003_test_rmse": None if fd003["test_rmse"] is None else _fmt(fd003["test_rmse"], 2),
        "fd001_baseline_constant_rmse": None
        if fd001.get("test_rmse_baseline_constant") is None
        else _fmt(fd001["test_rmse_baseline_constant"], 2),
        "fd003_baseline_constant_rmse": None
        if fd003.get("test_rmse_baseline_constant") is None
        else _fmt(fd003["test_rmse_baseline_constant"], 2),
        "alert_median_lead_cycles": _fmt(alert_stats["median_lead_time"], 2),
        "alert_engines_percent": _fmt(100.0 * lead_fraction, 2),
        "false_alerts_per_engine": _fmt(false_alerts, 4),
    }
    payload = {
        "project": "predictive-maintenance-ops",
        "success_criteria": success_criteria(
            fd001["test_rmse"],
            fd003["test_rmse"],
            lead_fraction,
            false_alerts,
            coverage,
        ),
        "metrics": {
            "datasets": ["FD001", "FD003"],
            "quick": quick,
            "fd001": fd001,
            "fd003": fd003,
            "alerts": {
                "lead_time_fraction": lead_fraction,
                "lead_time_percent": 100.0 * lead_fraction,
                "median_lead_time": alert_stats["median_lead_time"],
                "mean_lead_time": alert_stats["mean_lead_time"],
                "false_alerts_per_engine": false_alerts,
                "false_alert_cycles_per_engine": alert_stats["false_alert_cycles_per_engine"],
                "n_engines": alert_stats["n_engines"],
                "definition": (
                    "Lead time is true RUL at the first alert. "
                    "A false alert is an episode that starts while true RUL is above 50 cycles. "
                    "The reported rate is false episodes divided by validation engines."
                ),
            },
            "headline_text": headline,
        },
        "baseline": {
            "fd001": {
                "constant_mean_rul": base_by_name["FD001"]["constant_mean_rul"],
                "test_rmse_constant": fd001.get("test_rmse_baseline_constant"),
                "test_rmse_piecewise": fd001.get("test_rmse_baseline_piecewise"),
                "val_alert_lead_time_fraction": base_by_name["FD001"]["val_alert_lead_time_fraction"],
                "val_false_alerts_per_engine": base_by_name["FD001"]["val_false_alerts_per_engine"],
            },
            "fd003": {
                "constant_mean_rul": base_by_name["FD003"]["constant_mean_rul"],
                "test_rmse_constant": fd003.get("test_rmse_baseline_constant"),
                "test_rmse_piecewise": fd003.get("test_rmse_baseline_piecewise"),
                "val_alert_lead_time_fraction": base_by_name["FD003"]["val_alert_lead_time_fraction"],
                "val_false_alerts_per_engine": base_by_name["FD003"]["val_false_alerts_per_engine"],
            },
        },
        "splits": {
            info["dataset"]: {
                "train": int(info["train_engines"]),
                "val": int(info["val_engines"]),
                "test": int(info["test_engines"]),
            }
            for info in infos
        },
        "seeds": [0],
        "hardware": hardware,
        "runtime_seconds": runtime_seconds,
    }
    return payload, engines
