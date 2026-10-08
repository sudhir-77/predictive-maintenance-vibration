"""Time-domain stats plus FFT and envelope band energies."""

from __future__ import annotations

import numpy as np

from vibration.config import ENVELOPE_BAND_HZ, N_ENVELOPE_BANDS, N_FFT_BANDS
from vibration.envelope import envelope_spectrum, fault_orders


def rms(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(values * values)))


def excess_kurtosis(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    centered = values - np.mean(values)
    second = float(np.mean(centered * centered))
    if second <= 1e-18:
        return 0.0
    fourth = float(np.mean(centered**4))
    return fourth / (second * second) - 3.0


def crest_factor(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    scale = rms(values)
    if scale <= 1e-18:
        return 0.0
    return float(np.max(np.abs(values)) / scale)


def _band_fractions(freqs: np.ndarray, power: np.ndarray, edges: np.ndarray) -> list[float]:
    total = float(np.sum(power)) + 1e-12
    fractions: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        fractions.append(float(np.sum(power[mask]) / total))
    return fractions


def fft_band_energies(signal: np.ndarray, fs: float, n_bands: int = N_FFT_BANDS) -> list[float]:
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    centered = values - np.mean(values)
    window = np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(centered * window)) ** 2
    freqs = np.fft.rfftfreq(centered.size, d=1.0 / fs)
    edges = np.linspace(0.0, fs / 2.0, n_bands + 1)
    return _band_fractions(freqs, spectrum, edges)


def envelope_band_energies(signal: np.ndarray, fs: float, n_bands: int = N_ENVELOPE_BANDS) -> list[float]:
    freqs, spectrum = envelope_spectrum(signal, fs)
    power = spectrum**2
    edges = np.linspace(0.0, ENVELOPE_BAND_HZ, n_bands + 1)
    return _band_fractions(freqs, power, edges)


def order_peak_ratios(signal: np.ndarray, fs: float, rpm: float) -> list[float]:
    """Envelope amplitude at each fault order, divided by the spectral median."""
    if rpm <= 0:
        raise ValueError("rpm must be positive")
    freqs, spectrum = envelope_spectrum(signal, fs)
    shaft_hz = rpm / 60.0
    band = (freqs >= 10.0) & (freqs <= 500.0)
    floor = float(np.median(spectrum[band])) if np.any(band) else 0.0
    floor = max(floor, 1e-12)
    ratios: list[float] = []
    for _name, order in fault_orders():
        center = order * shaft_hz
        mask = np.abs(freqs - center) <= 2.0
        if not np.any(mask):
            ratios.append(0.0)
        else:
            ratios.append(float(np.max(spectrum[mask]) / floor))
    return ratios


def feature_names(n_fft: int = N_FFT_BANDS, n_env: int = N_ENVELOPE_BANDS) -> list[str]:
    names = ["rms", "kurtosis", "crest_factor"]
    names.extend(f"fft_band_{index}" for index in range(n_fft))
    names.extend(f"env_band_{index}" for index in range(n_env))
    names.extend(f"env_{name}" for name, _order in fault_orders())
    return names


def window_features(signal: np.ndarray, fs: float, rpm: float) -> np.ndarray:
    values = [
        rms(signal),
        excess_kurtosis(signal),
        crest_factor(signal),
        *fft_band_energies(signal, fs),
        *envelope_band_energies(signal, fs),
        *order_peak_ratios(signal, fs, rpm),
    ]
    return np.asarray(values, dtype=np.float64)
