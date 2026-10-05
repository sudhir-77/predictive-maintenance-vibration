"""Download verification without trusting a hand-typed metric."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from src.config import EXPECTED_ROWS
from src.download import (
    download_file,
    ensure_cmapss,
    extract_cmapss_zip,
    read_rul,
    read_telemetry,
    sha256_file,
    verify_extracted,
)


def test_sha256_and_verify_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "train_FD001.txt"
    path.write_text("1\n2\n", encoding="utf-8")
    digest = sha256_file(path)
    monkeypatch.setattr("src.download.EXPECTED_ROWS", {"train_FD001.txt": 2})
    monkeypatch.setattr("src.download.EXPECTED_SHA256", {"train_FD001.txt": digest})
    verify_extracted(tmp_path)
    path.write_text("1\n", encoding="utf-8")
    try:
        verify_extracted(tmp_path)
    except ValueError as exc:
        assert "rows" in str(exc)
    else:
        raise AssertionError("expected a row-count failure")


def test_nested_zip_extracts_expected_names(tmp_path):
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        for name in EXPECTED_ROWS:
            inner.writestr(name, "1 2 3\n")
    outer_path = tmp_path / "cmapss.zip"
    with zipfile.ZipFile(outer_path, "w") as outer:
        outer.writestr("6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip", inner_buffer.getvalue())
    dest = tmp_path / "out"
    extract_cmapss_zip(outer_path, dest)
    assert (dest / "train_FD001.txt").read_text(encoding="utf-8").strip() == "1 2 3"


def test_read_telemetry_assigns_source_row(tmp_path):
    columns = " ".join(str(i) for i in range(26))
    path = tmp_path / "tiny.txt"
    path.write_text(columns + "\n" + columns + "\n", encoding="utf-8")
    frame = read_telemetry(path)
    assert list(frame.columns[:2]) == ["unit_id", "cycle"]
    assert frame["source_row"].tolist() == [1, 2]
    assert frame.shape[1] == 27  # 26 measurements plus source_row


def test_read_rul(tmp_path):
    path = tmp_path / "RUL.txt"
    path.write_text("10\n\n12.5\n", encoding="utf-8")
    values = read_rul(path)
    assert values.tolist() == [10.0, 12.5]


def test_download_file_streams_bytes(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"abc"
            yield b"def"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("src.download.requests.get", lambda *args, **kwargs: Response())
    dest = tmp_path / "cmapss.zip"
    download_file("http://example.test/cmapss.zip", dest)
    assert dest.read_bytes() == b"abcdef"


def test_cached_cmapss_matches_checksums():
    root = Path(__file__).resolve().parents[1]
    dest = ensure_cmapss(root)
    verify_extracted(dest)
    train = pd.read_csv(dest / "train_FD001.txt", sep=r"\s+", header=None, engine="python")
    assert len(train) == 20631
