"""Download, verify, and parse NASA C-MAPSS text files."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import BASE_COLS, CMAPSS_URLS, EXPECTED_ROWS, EXPECTED_SHA256, OP_COLS, SENSOR_COLS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def verify_extracted(dest: Path) -> None:
    """Check row counts and SHA-256 of every extracted C-MAPSS file we use."""
    for name, rows in EXPECTED_ROWS.items():
        path = dest / name
        if not path.exists():
            raise FileNotFoundError(path)
        found = _nonempty_lines(path)
        if found != rows:
            raise ValueError(f"{name} has {found} rows, expected {rows}")
        digest = sha256_file(path)
        expected = EXPECTED_SHA256[name]
        if digest != expected:
            raise ValueError(f"{name} sha256 {digest} != {expected}")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "predictive-maintenance-ops/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)


def download_with_retries(dest: Path, attempts: int = 3) -> None:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        for url in CMAPSS_URLS:
            try:
                print(f"downloading C-MAPSS from {url} (attempt {attempt})")
                download_file(url, dest)
                return
            except Exception as exc:  # network or HTTP failure; try again
                last = exc
                print(f"download failed: {exc}")
    raise RuntimeError(f"could not download C-MAPSS: {last}")


def _extract_members(archive: zipfile.ZipFile, dest: Path) -> None:
    names = {Path(info.filename).name: info for info in archive.infolist()}
    missing = [name for name in EXPECTED_ROWS if name not in names]
    if missing:
        raise FileNotFoundError(f"zip is missing {missing}")
    dest.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_ROWS:
        dest.joinpath(name).write_bytes(archive.read(names[name]))


def extract_cmapss_zip(zip_path: Path, dest: Path) -> None:
    """Extract train/test/RUL files from the NASA zip (outer zip contains CMAPSSData.zip)."""
    with zipfile.ZipFile(zip_path) as outer:
        names = outer.namelist()
        flat = [name for name in names if Path(name).name in EXPECTED_ROWS]
        if len(flat) >= len(EXPECTED_ROWS):
            _extract_members(outer, dest)
            return
        inner_name = next(name for name in names if name.endswith("CMAPSSData.zip"))
        payload = outer.read(inner_name)
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        _extract_members(inner, dest)


def ensure_cmapss(root: Path) -> Path:
    """Return the cache directory, downloading and checking files when needed."""
    dest = root / "data" / "cmapss"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        verify_extracted(dest)
        return dest
    except (FileNotFoundError, ValueError) as exc:
        print(f"C-MAPSS cache incomplete ({exc}); extracting")
    zip_path = dest / "cmapss.zip"
    if not zip_path.exists():
        download_with_retries(zip_path)
    try:
        extract_cmapss_zip(zip_path, dest)
        verify_extracted(dest)
        return dest
    except Exception as exc:
        print(f"extract/verify failed ({exc}); downloading again")
        if zip_path.exists():
            zip_path.unlink()
        download_with_retries(zip_path)
        extract_cmapss_zip(zip_path, dest)
        verify_extracted(dest)
        return dest


def read_telemetry(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] < 26:
        raise ValueError(f"{path} has {frame.shape[1]} columns, expected 26")
    frame = frame.iloc[:, :26].copy()
    frame.columns = BASE_COLS
    frame["unit_id"] = frame["unit_id"].astype(np.int32)
    frame["cycle"] = frame["cycle"].astype(np.int32)
    for column in OP_COLS + SENSOR_COLS:
        frame[column] = frame[column].astype(np.float32)
    frame["source_row"] = np.arange(1, len(frame) + 1, dtype=np.int32)
    return frame


def read_rul(path: Path) -> np.ndarray:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            values.append(float(text.split()[0]))
    if not values:
        raise ValueError(f"{path} has no RUL values")
    return np.asarray(values, dtype=np.float64)
