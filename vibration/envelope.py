"""Bearing geometry and Hilbert envelope spectra."""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from vibration.config import (
    BALL_DIAMETER_IN,
    CONTACT_ANGLE_RAD,
    ENVELOPE_HIGH_HZ,
    ENVELOPE_LOW_HZ,
    N_BALLS,
    PITCH_DIAMETER_IN,
)


def geometry_ratio() -> float:
    return (BALL_DIAMETER_IN / PITCH_DIAMETER_IN) * math.cos(CONTACT_ANGLE_RAD)


def bpfo_hz(shaft_hz: float) -> float:
    """Ball-pass frequency of the outer race, in hertz, for a given shaft speed."""
    if shaft_hz <= 0:
        raise ValueError("shaft speed must be positive")
    return (N_BALLS / 2.0) * shaft_hz * (1.0 - geometry_ratio())


def bpfi_hz(shaft_hz: float) -> float:
    if shaft_hz <= 0:
        raise ValueError("shaft speed must be positive")
    return (N_BALLS / 2.0) * shaft_hz * (1.0 + geometry_ratio())


def bsf_hz(shaft_hz: float) -> float:
    """Ball spin frequency in hertz."""
    if shaft_hz <= 0:
        raise ValueError("shaft speed must be positive")
    ratio = geometry_ratio()
    return (PITCH_DIAMETER_IN / (2.0 * BALL_DIAMETER_IN)) * shaft_hz * (1.0 - ratio * ratio)


def fault_orders() -> list[tuple[str, float]]:
    """Orders (cycles per revolution), independent of the motor load."""
    return [
        ("shaft", 1.0),
        ("bpfo", bpfo_hz(1.0)),
        ("bpfo_2", 2.0 * bpfo_hz(1.0)),
        ("bpfi", bpfi_hz(1.0)),
        ("bpfi_2", 2.0 * bpfi_hz(1.0)),
        ("bsf", bsf_hz(1.0)),
        ("bsf_2", 2.0 * bsf_hz(1.0)),
    ]


def outer_race_search_band(shaft_hz: float) -> tuple[float, float]:
    """A wide band around outer-race orders, not centered on the theoretical BPFO.

    The lower edge sits above two times running speed. The upper edge stays below
    the inner-race order (about 5.4 times running speed). BPFO for this bearing
    is near 3.6 times running speed, so it falls inside the band without the
    detector being told that number.
    """
    if shaft_hz <= 0:
        raise ValueError("shaft speed must be positive")
    return (2.5 * shaft_hz, 5.0 * shaft_hz)


def bandpass(signal: np.ndarray, fs: float, low_hz: float = ENVELOPE_LOW_HZ, high_hz: float = ENVELOPE_HIGH_HZ) -> np.ndarray:
    if not (0 < low_hz < high_hz < fs / 2):
        raise ValueError(f"band {low_hz}-{high_hz} Hz is outside Nyquist {fs / 2}")
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))


def envelope_spectrum(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert envelope spectrum after a resonance bandpass. DC is removed."""
    filtered = bandpass(signal, fs)
    envelope = np.abs(hilbert(filtered))
    envelope = envelope - float(np.mean(envelope))
    window = np.hanning(envelope.size)
    spectrum = np.abs(np.fft.rfft(envelope * window))
    freqs = np.fft.rfftfreq(envelope.size, d=1.0 / fs)
    return freqs, spectrum


def _parabolic_peak(freqs: np.ndarray, spectrum: np.ndarray, index: int) -> float:
    if index <= 0 or index >= spectrum.size - 1:
        return float(freqs[index])
    left, center, right = float(spectrum[index - 1]), float(spectrum[index]), float(spectrum[index + 1])
    denom = left - 2.0 * center + right
    if denom == 0.0:
        return float(freqs[index])
    delta = 0.5 * (left - right) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0
    return float(freqs[index] + delta * df)


def detect_envelope_peak(signal: np.ndarray, fs: float, f_low: float, f_high: float) -> float:
    """Frequency of the envelope peak in [f_low, f_high], reinforced by its second harmonic."""
    freqs, spectrum = envelope_spectrum(signal, fs)
    if f_low >= f_high:
        raise ValueError("search band is empty")
    candidates: list[int] = []
    for index in range(1, spectrum.size - 1):
        freq = float(freqs[index])
        if freq < f_low or freq > f_high:
            continue
        if spectrum[index] >= spectrum[index - 1] and spectrum[index] >= spectrum[index + 1]:
            candidates.append(index)
    if not candidates:
        band = np.flatnonzero((freqs >= f_low) & (freqs <= f_high))
        if band.size == 0:
            raise ValueError("search band contains no frequency bins")
        return _parabolic_peak(freqs, spectrum, int(band[np.argmax(spectrum[band])]))
    df = float(freqs[1] - freqs[0])

    def score(index: int) -> float:
        harmonic = int(round(float(freqs[index]) * 2.0 / df)) if df > 0 else index
        harmonic_amp = float(spectrum[harmonic]) if 0 <= harmonic < spectrum.size else 0.0
        return float(spectrum[index]) + 0.5 * harmonic_amp

    best = max(candidates, key=score)
    return _parabolic_peak(freqs, spectrum, best)


def outer_race_check(signal: np.ndarray, fs: float, rpm: float) -> dict[str, float]:
    shaft_hz = rpm / 60.0
    theory = bpfo_hz(shaft_hz)
    f_low, f_high = outer_race_search_band(shaft_hz)
    detected = detect_envelope_peak(signal, fs, f_low, f_high)
    rel_error = abs(detected - theory) / theory
    return {
        "bpfo_hz_theory": theory,
        "bpfo_hz_detected": detected,
        "rel_error": rel_error,
        "shaft_hz": shaft_hz,
        "search_low_hz": f_low,
        "search_high_hz": f_high,
        "rpm": float(rpm),
    }
