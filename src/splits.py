"""Engine-level train, validation, and official-test assignments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import SPLIT_VERSION


def assign_units(unit_ids: list[int] | np.ndarray, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Hold out a seeded fraction of engine ids for validation. The rest stay in train."""
    units = np.array(sorted(int(unit) for unit in unit_ids), dtype=np.int32)
    if len(units) < 2:
        raise ValueError("need at least two engines to form a train and validation split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(units)
    n_val = int(round(len(units) * val_fraction))
    n_val = min(max(n_val, 1), len(units) - 1)
    val_units = sorted(int(unit) for unit in perm[:n_val])
    train_units = sorted(int(unit) for unit in perm[n_val:])
    return train_units, val_units


def _valid_payload(payload: dict, dataset: str, quick: bool, train_ids: set[int], test_ids: set[int]) -> bool:
    if payload.get("version") != SPLIT_VERSION:
        return False
    if payload.get("dataset") != dataset or bool(payload.get("quick")) != bool(quick):
        return False
    try:
        train = {int(unit) for unit in payload["train_units"]}
        val = {int(unit) for unit in payload["val_units"]}
        test = {int(unit) for unit in payload["test_units"]}
    except (KeyError, TypeError, ValueError):
        return False
    if train & val:
        return False
    if train | val != train_ids:
        return False
    if test != test_ids:
        return False
    return True


def load_or_create_split(
    path: Path,
    dataset: str,
    train_unit_ids: list[int],
    test_unit_ids: list[int],
    val_fraction: float,
    seed: int,
    quick: bool,
) -> dict:
    """Load a saved split when it matches this dataset, otherwise create and save one."""
    train_ids = {int(unit) for unit in train_unit_ids}
    test_ids = {int(unit) for unit in test_unit_ids}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _valid_payload(payload, dataset, quick, train_ids, test_ids):
            return payload
    train_units, val_units = assign_units(sorted(train_ids), val_fraction, seed)
    payload = {
        "version": SPLIT_VERSION,
        "dataset": dataset,
        "quick": bool(quick),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_units": train_units,
        "val_units": val_units,
        "test_units": sorted(test_ids),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
