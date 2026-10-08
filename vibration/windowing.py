"""Cut windows inside one recording and keep every recording in a single split."""

from __future__ import annotations

from vibration.catalog import Recording
from vibration.config import CLASS_ORDER, DEV_LOADS, TEST_LOADS, VAL_CANDIDATE_LOADS


def window_length_for_rpm(rpm: float, sample_rate_hz: float, n_revs: int) -> int:
    if rpm <= 0 or n_revs < 1:
        raise ValueError("rpm and n_revs must be positive")
    length = int(round(n_revs * sample_rate_hz / (rpm / 60.0)))
    return max(length, 8)


def split_for_load(load_hp: int) -> str:
    """Return the split for a load that is not shared between train and validation."""
    if load_hp in TEST_LOADS:
        return "test"
    raise ValueError(f"load {load_hp} hp is split by recording, not by the whole load")


def split_for_recording(label: str, load_hp: int) -> str:
    """Put each development file in train or validation, and the test load aside."""
    if load_hp in TEST_LOADS:
        return "test"
    if load_hp not in DEV_LOADS:
        raise ValueError(f"load {load_hp} hp is not in the train/val/test plan")
    if label not in CLASS_ORDER:
        raise ValueError(f"unknown fault class {label}")
    held_out = VAL_CANDIDATE_LOADS[CLASS_ORDER.index(label) % len(VAL_CANDIDATE_LOADS)]
    if load_hp == held_out:
        return "val"
    return "train"


def window_starts(n_samples: int, length: int, n_windows: int) -> list[int]:
    """Evenly spaced starts. Windows may overlap inside one recording and one split."""
    if length < 8:
        raise ValueError("window length is too short")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    last = n_samples - length
    if last < 0:
        raise ValueError(f"signal length {n_samples} is shorter than window {length}")
    if n_windows == 1 or last == 0:
        return [0]
    step = last / (n_windows - 1)
    starts = [int(round(index * step)) for index in range(n_windows)]
    starts = [min(max(start, 0), last) for start in starts]
    unique = sorted(set(starts))
    return unique


def ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Half-open ranges [start, end) share a sample."""
    return start_a < end_b and start_b < end_a


def assign_window_splits(
    n_samples: int,
    length: int,
    n_windows: int,
    load_hp: int,
    label: str,
) -> list[list[int | str]]:
    """Place every window from one recording into that recording's single split."""
    split = split_for_recording(label, load_hp)
    starts = window_starts(n_samples, length, n_windows)
    rows = [[start, start + length, split] for start in starts]
    if not rows:
        raise ValueError(f"no windows fit in a {n_samples}-sample recording")
    return rows


def cross_split_overlap_count(document: dict) -> int:
    """Count window pairs from the same recording that sit in different splits and share samples."""
    count = 0
    grouped: dict[str, list[tuple[str, int, int]]] = {}
    for file_id, info in document["recordings"].items():
        for window in info["windows"]:
            if len(window) == 3:
                start, end, split = window
            else:
                start, end = window
                split = info["split"]
            grouped.setdefault(file_id, []).append((str(split), int(start), int(end)))
    for windows in grouped.values():
        for index, (split_a, start_a, end_a) in enumerate(windows):
            for split_b, start_b, end_b in windows[index + 1 :]:
                if split_a != split_b and ranges_overlap(start_a, end_a, start_b, end_b):
                    count += 1
    return count


def build_split_document(
    recordings: list[Recording] | tuple[Recording, ...],
    lengths: dict[str, int],
    measured_rpm: dict[str, float | None],
    window_length: int | None,
    n_windows: int,
    sample_rate_hz: int,
    version: int,
    n_revs: int | None = None,
) -> dict:
    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    details: dict[str, dict] = {}
    for spec in recordings:
        n_samples = int(lengths[spec.file_id])
        rpm = measured_rpm.get(spec.file_id)
        rpm_value = float(rpm) if rpm is not None else float(spec.nominal_rpm)
        if n_revs is not None:
            length = window_length_for_rpm(rpm_value, sample_rate_hz, n_revs)
        else:
            if window_length is None:
                raise ValueError("window_length is required when n_revs is not set")
            length = int(window_length)
        windows = assign_window_splits(n_samples, length, n_windows, spec.load_hp, spec.label)
        present = sorted({str(row[2]) for row in windows})
        if len(present) != 1:
            raise ValueError(f"recording {spec.file_id} spans splits {present}")
        by_split[present[0]].append(spec.file_id)
        details[spec.file_id] = {
            "load_hp": spec.load_hp,
            "label": spec.label,
            "family": spec.family,
            "nominal_rpm": spec.nominal_rpm,
            "measured_rpm": None if rpm is None else float(rpm),
            "n_samples": n_samples,
            "window_length": length,
            "split": present[0],
            "windows": windows,
        }
    train_ids = set(by_split["train"])
    val_ids = set(by_split["val"])
    test_ids = set(by_split["test"])
    if (
        len(train_ids) != len(by_split["train"])
        or len(val_ids) != len(by_split["val"])
        or len(test_ids) != len(by_split["test"])
        or not train_ids.isdisjoint(val_ids)
        or not train_ids.isdisjoint(test_ids)
        or not val_ids.isdisjoint(test_ids)
    ):
        raise ValueError("train, validation, and test recordings must be disjoint")
    return {
        "version": version,
        "sample_rate_hz": sample_rate_hz,
        "window_length": window_length,
        "n_revs": n_revs,
        "train_recordings": by_split["train"],
        "val_recordings": by_split["val"],
        "test_recordings": by_split["test"],
        "recordings": details,
    }
