"""Load the drive-end accelerometer channel from a CWRU .mat file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io


def load_drive_end(path: Path) -> tuple[np.ndarray, float | None]:
    """Return the DE time series as float64 and the recorded RPM when the file has it."""
    mat = scipy.io.loadmat(path)
    de_key = next((key for key in mat if str(key).endswith("_DE_time")), None)
    if de_key is None:
        raise KeyError(f"{path.name} has no drive-end time series")
    signal = np.asarray(mat[de_key], dtype=np.float64).reshape(-1)
    if signal.size < 2 or not np.isfinite(signal).all():
        raise ValueError(f"{path.name} drive-end signal is empty or non-finite")
    rpm: float | None = None
    for key in mat:
        if str(key).upper().endswith("RPM"):
            values = np.asarray(mat[key], dtype=np.float64).reshape(-1)
            if values.size and np.isfinite(values[0]) and values[0] > 0:
                rpm = float(values[0])
                break
    return signal, rpm
