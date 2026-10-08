"""Spectra and confusion matrices for the vibration layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from vibration.envelope import bpfo_hz, envelope_spectrum


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(y_true: list[str], y_pred: list[str], labels: list[str], path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    _save(fig, path)


def plot_envelope(signal: np.ndarray, fs: float, rpm: float, detected_hz: float, path: Path) -> None:
    freqs, spectrum = envelope_spectrum(signal, fs)
    theory = bpfo_hz(rpm / 60.0)
    limit = 300.0
    mask = freqs <= limit
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(freqs[mask], spectrum[mask], color="#1f4e79", linewidth=1.0)
    ax.axvline(theory, color="#b85c38", linestyle="--", label=f"BPFO theory {theory:.2f} Hz")
    ax.axvline(detected_hz, color="#2e7d32", linestyle=":", label=f"detected {detected_hz:.2f} Hz")
    ax.set_xlabel("Envelope frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Outer-race envelope spectrum")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_window_spectra(normal: np.ndarray, fault: np.ndarray, fs: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for values, name, color in ((normal, "normal", "#1f4e79"), (fault, "outer race", "#b85c38")):
        centered = values - np.mean(values)
        spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
        freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs)
        mask = freqs <= 2000
        ax.plot(freqs[mask], spectrum[mask], label=name, color=color, linewidth=1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Drive-end FFT, one window")
    ax.legend(frameon=False)
    _save(fig, path)
