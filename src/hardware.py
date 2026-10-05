"""Host info and a lightweight peak-RSS sampler."""

from __future__ import annotations

import platform
import threading

import psutil

from src.config import N_THREADS

_TORCH_CONFIGURED = False


def configure_torch_threads() -> None:
    """Cap PyTorch threads. Inter-op threads can only be set once per process."""
    global _TORCH_CONFIGURED
    import torch

    torch.set_num_threads(N_THREADS)
    if not _TORCH_CONFIGURED:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _TORCH_CONFIGURED = True


def hardware_dict() -> dict:
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = None
    cpu = platform.processor() or platform.machine()
    return {
        "cpu": cpu,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": gpu,
        "python": platform.python_version(),
        "os": platform.platform(),
    }


def _process_rss(proc: psutil.Process) -> int:
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            continue
    return total


class PeakRSS:
    """Sample this process and its children until stop()."""

    def __init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="peak-rss", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        proc = psutil.Process()
        while True:
            self.peak_bytes = max(self.peak_bytes, _process_rss(proc))
            if self._stop.wait(0.25):
                break
        self.peak_bytes = max(self.peak_bytes, _process_rss(proc))

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.gb

    @property
    def gb(self) -> float:
        return self.peak_bytes / (1024**3)
