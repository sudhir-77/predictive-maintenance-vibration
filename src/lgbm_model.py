"""LightGBM remaining-useful-life regressor."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import LGBM_EARLY_STOPPING, LGBM_MAX_ROUNDS, N_THREADS, RUL_CLIP, SEED


def lgbm_params(seed: int = SEED, min_child_samples: int = 20) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": min_child_samples,
        "lambda_l2": 0.1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": N_THREADS,
    }


def sample_weights(rul_true: np.ndarray, clip: int = RUL_CLIP) -> np.ndarray:
    """Upweight cycles as remaining life falls, so the flat early plateau does not dominate."""
    remaining = np.asarray(rul_true, dtype=np.float64)
    closeness = np.clip((clip - remaining) / float(clip), 0.0, 1.0)
    return 1.0 + 2.0 * closeness


@dataclass
class TrainedLGBM:
    booster: lgb.Booster
    feature_cols: list[str]
    num_iteration: int


def train_lgbm(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "rul_target",
    val_df: pd.DataFrame | None = None,
    max_rounds: int = LGBM_MAX_ROUNDS,
    early_stopping: int = LGBM_EARLY_STOPPING,
    seed: int = SEED,
    fixed_rounds: int | None = None,
    min_child_samples: int = 20,
) -> TrainedLGBM:
    if (train_df["split"] == "test").any() or (train_df["origin"] == "test_file").any():
        raise RuntimeError("test rows leaked into LightGBM training")
    if any("rul" in column for column in feature_cols):
        raise RuntimeError(f"label-like feature columns: {feature_cols}")
    train_x = train_df[feature_cols]
    train_y = train_df[target_col].to_numpy(dtype=np.float64)
    weights = sample_weights(train_df["rul_true"].to_numpy(dtype=np.float64))
    dtrain = lgb.Dataset(train_x, label=train_y, weight=weights, feature_name=feature_cols, free_raw_data=False)
    rounds = max_rounds if fixed_rounds is None else int(fixed_rounds)
    callbacks = [lgb.log_evaluation(period=0)]
    valid_sets = None
    valid_names = None
    if fixed_rounds is None and val_df is not None and len(val_df) > 0:
        if (val_df["origin"] == "test_file").any():
            raise RuntimeError("test rows leaked into LightGBM validation")
        dval = lgb.Dataset(
            val_df[feature_cols],
            label=val_df[target_col].to_numpy(dtype=np.float64),
            reference=dtrain,
            free_raw_data=False,
        )
        valid_sets = [dval]
        valid_names = ["val"]
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping, verbose=False))
    booster = lgb.train(
        lgbm_params(seed, min_child_samples=min_child_samples),
        dtrain,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    if fixed_rounds is None:
        best = int(booster.best_iteration or rounds)
    else:
        best = int(fixed_rounds)
    return TrainedLGBM(booster=booster, feature_cols=list(feature_cols), num_iteration=best)


def predict_lgbm(model: TrainedLGBM, frame: pd.DataFrame) -> np.ndarray:
    pred = model.booster.predict(frame[model.feature_cols], num_iteration=model.num_iteration)
    return np.asarray(pred, dtype=np.float64)
