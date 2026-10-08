"""Download CWRU .mat files into data/cwru and check SHA-256."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

from vibration.catalog import Recording
from vibration.config import CWRU_URL

_CHECKSUM_PATH = Path(__file__).with_name("checksums.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums() -> dict[str, str]:
    payload = json.loads(_CHECKSUM_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checksums.json must be an object")
    return {str(key): str(value) for key, value in payload.items()}


def verify_checksum(path: Path, file_id: str, checksums: dict[str, str] | None = None) -> str:
    expected_map = load_checksums() if checksums is None else checksums
    if file_id not in expected_map:
        raise KeyError(f"no checksum for recording {file_id}")
    found = sha256_file(path)
    expected = expected_map[file_id]
    if found != expected:
        raise ValueError(f"{file_id}.mat sha256 {found} != {expected}")
    return found


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "predictive-maintenance-vibration/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)


def ensure_recordings(root: Path, recordings: tuple[Recording, ...] | list[Recording]) -> Path:
    """Cache each recording under data/cwru. Existing files are re-checked, not re-downloaded."""
    dest = root / "data" / "cwru"
    dest.mkdir(parents=True, exist_ok=True)
    checksums = load_checksums()
    for spec in recordings:
        path = dest / f"{spec.file_id}.mat"
        if path.exists():
            try:
                verify_checksum(path, spec.file_id, checksums)
                continue
            except ValueError:
                path.unlink()
        last: Exception | None = None
        for attempt in range(1, 4):
            try:
                url = CWRU_URL.format(file_id=spec.file_id)
                print(f"downloading CWRU {spec.file_id}.mat (attempt {attempt})")
                download_file(url, path)
                verify_checksum(path, spec.file_id, checksums)
                last = None
                break
            except Exception as exc:
                last = exc
                if path.exists():
                    path.unlink()
                print(f"download failed for {spec.file_id}: {exc}")
        if last is not None:
            raise RuntimeError(f"could not download {spec.file_id}.mat: {last}") from last
    return dest
