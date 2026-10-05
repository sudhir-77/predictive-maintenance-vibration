"""Fit one C-MAPSS subset: features, models, validation alerts, then one test score."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.alerts import apply_policy, tune_policy
from src.anomaly import fit_anomaly, score_anomaly
from src.baseline import (
    fit_piecewise,
    fit_sensor_threshold,
    predict_constant,
    predict_piecewise,
    sensor_threshold_alerts,
)
from src.cnn_model import predict_cnn, resolve_device, train_cnn
from src.config import (
    CNN_EPOCHS,
    CNN_PATIENCE,
    CNN_WINDOW,
    FILES,
    IFOREST_TREES,
    LGBM_EARLY_STOPPING,
    LGBM_MAX_ROUNDS,
    QUICK_CNN_EPOCHS,
    QUICK_DEV_UNITS,
    QUICK_IFOREST_TREES,
    QUICK_LGBM_EARLY_STOPPING,
    QUICK_LGBM_ROUNDS,
    QUICK_TEST_UNITS,
    RUL_CLIP,
    SEED,
    VAL_FRACTION,
)
from src.download import read_rul, read_telemetry
from src.features import build_feature_table, cnn_feature_names, tabular_feature_names
from src.lgbm_model import predict_lgbm, train_lgbm
from src.metrics import last_cycle_rows, rmse, score_alert_arrays
from src.selection import apply_chosen, choose_alpha, choose_calibrator
from src.splits import load_or_create_split


@dataclass
class DatasetResult:
    test_predictions: pd.DataFrame
    val_cycles: pd.DataFrame
    info: dict
    baseline: dict


def _take_units(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    keep = sorted(int(unit) for unit in frame["unit_id"].unique())[:limit]
    return frame[frame["unit_id"].isin(keep)].copy()


def _life_rul(frame: pd.DataFrame) -> pd.Series:
    maximum = frame.groupby("engine_key")["cycle"].transform("max")
    return (maximum - frame["cycle"]).astype(np.float64)


def _assert_no_test(frame: pd.DataFrame, what: str) -> None:
    if frame.empty:
        return
    if (frame["origin"] == "test_file").any() or (frame["split"] == "test").any():
        raise RuntimeError(f"{what} includes official test rows")


def evaluate_dataset(
    data_dir: Path,
    splits_dir: Path,
    dataset: str,
    quick: bool,
    device_name: str,
    score_test: bool,
) -> DatasetResult:
    print(f"=== {dataset} quick={quick} score_test={score_test} ===")
    names = FILES[dataset]
    train_raw = read_telemetry(data_dir / names["train"])
    test_raw = read_telemetry(data_dir / names["test"])
    full_test_units = sorted(int(unit) for unit in test_raw["unit_id"].unique())
    if full_test_units != list(range(1, len(full_test_units) + 1)):
        raise RuntimeError(f"{dataset} test unit ids are not 1..n")
    full_test_count = len(full_test_units)
    if quick:
        train_raw = _take_units(train_raw, QUICK_DEV_UNITS)
        test_raw = _take_units(test_raw, QUICK_TEST_UNITS)

    dev_units = sorted(int(unit) for unit in train_raw["unit_id"].unique())
    test_units = sorted(int(unit) for unit in test_raw["unit_id"].unique())
    split_path = splits_dir / (f"{dataset}-quick.json" if quick else f"{dataset}.json")
    split = load_or_create_split(
        split_path,
        dataset,
        dev_units,
        test_units,
        VAL_FRACTION,
        SEED,
        quick,
    )
    train_units = set(int(unit) for unit in split["train_units"])
    val_units = set(int(unit) for unit in split["val_units"])

    def _decorate(raw: pd.DataFrame, origin: str) -> pd.DataFrame:
        frame = raw.copy()
        frame["dataset"] = dataset
        frame["origin"] = origin
        frame["engine_key"] = origin + "-" + frame["unit_id"].astype(str)
        if origin == "train_file":
            frame["split"] = np.where(frame["unit_id"].isin(val_units), "val", "train")
            frame["rul_true"] = _life_rul(frame)
        else:
            frame["split"] = "test"
            frame["rul_true"] = np.nan
        return frame

    dev = _decorate(train_raw, "train_file")
    test = _decorate(test_raw, "test_file") if score_test else dev.iloc[0:0].copy()
    if score_test:
        combined = pd.concat([dev, test], ignore_index=True)
    else:
        combined = dev
    train_mask = combined["split"].eq("train")
    print(f"  rows train={int(train_mask.sum())} val={int(combined['split'].eq('val').sum())} test={int(combined['split'].eq('test').sum())}")
    featured, sensors, _signs = build_feature_table(combined, train_mask)
    feature_cols = tabular_feature_names(sensors)
    cnn_cols = cnn_feature_names(sensors)
    if any("rul" in column for column in feature_cols):
        raise RuntimeError("feature list contains a label column")

    train_df = featured[featured["split"].eq("train")].copy()
    val_df = featured[featured["split"].eq("val")].copy()
    test_df = featured[featured["split"].eq("test")].copy()
    _assert_no_test(train_df, "train")
    _assert_no_test(val_df, "val")

    max_rounds = QUICK_LGBM_ROUNDS if quick else LGBM_MAX_ROUNDS
    early = QUICK_LGBM_EARLY_STOPPING if quick else LGBM_EARLY_STOPPING
    cnn_epochs = QUICK_CNN_EPOCHS if quick else CNN_EPOCHS
    n_trees = QUICK_IFOREST_TREES if quick else IFOREST_TREES
    device = resolve_device(device_name)
    # Early stopping uses cycles at or below the training clip. Above the clip the
    # target is flat and would hide error in the degradation region.
    val_stop = val_df[val_df["rul_true"] <= RUL_CLIP]
    if val_stop.empty:
        val_stop = val_df

    print("  training LightGBM on train engines")
    lgbm = train_lgbm(train_df, feature_cols, val_df=val_stop, max_rounds=max_rounds, early_stopping=early, seed=SEED)
    print(f"  lightgbm best_iteration={lgbm.num_iteration}")
    print("  training 1D-CNN on train engines")
    cnn, cnn_epochs_used = train_cnn(
        train_df,
        cnn_cols,
        device,
        val_df=val_df,
        window=CNN_WINDOW,
        epochs=cnn_epochs,
        patience=CNN_PATIENCE if not quick else 1,
        seed=SEED,
    )

    raw_lgb_val = predict_lgbm(lgbm, val_df)
    raw_cnn_val = predict_cnn(cnn, val_df, cnn_cols, device, window=CNN_WINDOW, last_only=False)
    if not np.isfinite(raw_lgb_val).all() or not np.isfinite(raw_cnn_val).all():
        raise RuntimeError(f"{dataset} validation predictions are not finite")
    val_df = val_df.copy()
    val_df["raw_lgb"] = raw_lgb_val
    val_df["raw_cnn"] = raw_cnn_val
    # Calibrate where the training target is meaningful, plus a short band above the
    # clip. Cycles far above that have a flat sensor trace and a line fit there
    # stretches forecasts in the failure region.
    selectable = val_df[val_df["rul_true"] <= (RUL_CLIP + 25)]
    if len(selectable) < 30:
        selectable = val_df
    cal_lgb, _ = choose_calibrator(
        selectable["rul_true"].to_numpy(),
        selectable["raw_lgb"].to_numpy(),
        selectable["engine_key"].to_numpy(),
    )
    cal_cnn, _ = choose_calibrator(
        selectable["rul_true"].to_numpy(),
        selectable["raw_cnn"].to_numpy(),
        selectable["engine_key"].to_numpy(),
    )
    cal_lgb_sel = apply_chosen(selectable["raw_lgb"].to_numpy(), cal_lgb)
    cal_cnn_sel = apply_chosen(selectable["raw_cnn"].to_numpy(), cal_cnn)
    alpha, val_rmse = choose_alpha(selectable["rul_true"].to_numpy(), cal_lgb_sel, cal_cnn_sel)
    print(
        f"  val identifiable RMSE={val_rmse:.3f} alpha_lgbm={alpha} "
        f"cal_lgb={cal_lgb.kind}({cal_lgb.intercept:.3f},{cal_lgb.slope:.3f}) "
        f"cal_cnn={cal_cnn.kind}({cal_cnn.intercept:.3f},{cal_cnn.slope:.3f})"
    )

    val_df["rul_pred_lgbm"] = apply_chosen(val_df["raw_lgb"].to_numpy(), cal_lgb)
    val_df["rul_pred_cnn"] = apply_chosen(val_df["raw_cnn"].to_numpy(), cal_cnn)
    val_df["rul_pred"] = alpha * val_df["rul_pred_lgbm"] + (1.0 - alpha) * val_df["rul_pred_cnn"]
    proxy = val_df[val_df["rul_true"] <= (RUL_CLIP + 25)]
    if len(proxy):
        print(f"  val RMSE on cycles with RUL<={RUL_CLIP + 25}={rmse(proxy['rul_true'], proxy['rul_pred']):.3f}")

    anomaly_cols = [f"{sensor}_roll_mean_10" for sensor in sensors]
    anomaly = fit_anomaly(train_df, anomaly_cols, seed=SEED, n_estimators=n_trees, n_jobs=4 if not quick else 1)
    control_z, iforest = score_anomaly(anomaly, val_df)
    val_df["control_chart_z"] = control_z
    val_df["iforest_score"] = iforest
    policy, policy_stats = tune_policy(val_df, anomaly.score_thresholds, quick=quick)
    val_df["alert"] = apply_policy(val_df, policy).astype(int)
    val_df["anomaly_flag"] = (
        (val_df["control_chart_z"] >= policy.z_threshold) | (val_df["iforest_score"] >= policy.iforest_threshold)
    ).astype(int)
    print(
        f"  alerts lead_fraction={policy_stats['lead_time_fraction']:.3f} "
        f"false_per_engine={policy_stats['false_alerts_per_engine']:.3f} "
        f"passed={policy_stats['passed_constraints']} policy={policy.as_dict()}"
    )

    sensor_rule = fit_sensor_threshold(train_df, sensors)
    scored_val = val_df.copy()
    scored_val["baseline_alert"] = sensor_threshold_alerts(val_df, sensor_rule).to_numpy()
    baseline_flags: list[np.ndarray] = []
    baseline_ruls: list[np.ndarray] = []
    for _, group in scored_val.groupby("engine_key", sort=False):
        ordered = group.sort_values("cycle")
        baseline_flags.append(ordered["baseline_alert"].to_numpy(dtype=bool))
        baseline_ruls.append(ordered["rul_true"].to_numpy(dtype=np.float64))
    baseline_alert_stats = score_alert_arrays(baseline_flags, baseline_ruls)

    test_predictions = pd.DataFrame()
    baseline_model = None
    if score_test:
        print("  refitting on all development engines for the official test score")
        dev_df = featured[featured["split"].isin(["train", "val"])].copy()
        _assert_no_test(dev_df, "refit")
        lgbm_full = train_lgbm(
            dev_df,
            feature_cols,
            val_df=None,
            fixed_rounds=lgbm.num_iteration,
            seed=SEED,
        )
        cnn_full, _ = train_cnn(
            dev_df,
            cnn_cols,
            device,
            val_df=None,
            window=CNN_WINDOW,
            epochs=max(cnn_epochs_used, 1),
            patience=CNN_PATIENCE,
            seed=SEED,
        )
        # Score every test cycle, then keep the last cycle. The CNN window needs the earlier cycles.
        test_scored = test_df.copy()
        test_scored["rul_pred_lgbm"] = apply_chosen(predict_lgbm(lgbm_full, test_scored), cal_lgb)
        test_scored["rul_pred_cnn"] = apply_chosen(
            predict_cnn(cnn_full, test_scored, cnn_cols, device, window=CNN_WINDOW, last_only=False),
            cal_cnn,
        )
        test_scored["rul_pred"] = alpha * test_scored["rul_pred_lgbm"] + (1.0 - alpha) * test_scored["rul_pred_cnn"]
        test_last = last_cycle_rows(test_scored)
        # Official RUL file is read only here, after thresholds and blend weights are fixed.
        rul_values = read_rul(data_dir / names["rul"])
        if len(rul_values) != full_test_count:
            raise RuntimeError(f"{dataset} RUL file length {len(rul_values)} != {full_test_count} test engines")
        test_last["rul_true"] = test_last["unit_id"].map(lambda unit: float(rul_values[int(unit) - 1]))
        if test_last["rul_true"].isna().any():
            raise RuntimeError("failed to align official RUL labels")
        baseline_model = fit_piecewise(dev_df, sensors)
        test_last["rul_pred_baseline_constant"] = predict_constant(baseline_model, len(test_last))
        test_last["rul_pred_baseline_piecewise"] = predict_piecewise(baseline_model, test_last)
        test_predictions = test_last
        print(
            f"  test RMSE selected={rmse(test_last['rul_true'], test_last['rul_pred']):.3f} "
            f"lgbm={rmse(test_last['rul_true'], test_last['rul_pred_lgbm']):.3f} "
            f"cnn={rmse(test_last['rul_true'], test_last['rul_pred_cnn']):.3f}"
        )

    dev_for_baseline = featured[featured["split"].isin(["train", "val"])].copy()
    if baseline_model is None:
        baseline_model = fit_piecewise(dev_for_baseline, sensors)

    info = {
        "dataset": dataset,
        "sensors": sensors,
        "n_features": len(feature_cols),
        "lgbm_iterations": lgbm.num_iteration,
        "cnn_epochs": cnn_epochs_used,
        "blend_alpha_lgbm": alpha,
        "calibrator_lgbm": cal_lgb.as_dict(),
        "calibrator_cnn": cal_cnn.as_dict(),
        "val_truncated_rmse": val_rmse,
        "alert_policy": policy.as_dict(),
        "alert_tune": policy_stats,
        "train_engines": len(train_units),
        "val_engines": len(val_units),
        "test_engines": len(test_units) if score_test else 0,
        "quick": quick,
    }
    baseline = {
        "dataset": dataset,
        "constant_mean_rul": baseline_model.constant,
        "piecewise_sensor": baseline_model.sensor,
        "piecewise_intercept": baseline_model.intercept,
        "piecewise_coef": baseline_model.coef,
        "sensor_threshold": {
            "sensor": sensor_rule.sensor,
            "column": sensor_rule.column,
            "threshold": sensor_rule.threshold,
            "direction": sensor_rule.direction,
            "persist": sensor_rule.persist,
        },
        "val_alert_lead_time_fraction": baseline_alert_stats["lead_time_fraction"],
        "val_false_alerts_per_engine": baseline_alert_stats["false_alerts_per_engine"],
        "val_median_lead_time": baseline_alert_stats["median_lead_time"],
    }
    return DatasetResult(test_predictions=test_predictions, val_cycles=val_df, info=info, baseline=baseline)
