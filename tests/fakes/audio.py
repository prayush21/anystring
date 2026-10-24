"""Synthetic mic signals: silence, room noise, string brushes and speaker bleed.

All signals are mono float32 in [-1, 1]. Every generator that places events also
returns their true onset times in seconds, for use as ground truth.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

SR = 48_000


def silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(round(duration_s * sr), dtype=np.float32)


def room_noise(
    duration_s: float, *, rms: float = 0.003, sr: int = SR, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Pinkish background noise (1/f tilt, most energy below 1 kHz), scaled to ``rms``."""
    rng = rng or np.random.default_rng(0)
    white = rng.normal(0.0, 1.0, round(duration_s * sr))
    # Paul Kellet's economy pink filter.
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = sps.lfilter(b, a, white)
    return _scale_rms(pink, rms)


def brush_strokes(
    onsets_s: Sequence[float],
    duration_s: float,
    *,
    peak: float = 0.2,
    decay_ms: float = 25.0,
    sr: int = SR,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """A fingertip brushing racket strings: short high-band noise bursts.

    Each burst has a 0.5 ms attack and an exponential decay, band-passed to 2..8 kHz.
    Returns ``(signal, onset_times_s)``.
    """
    rng = rng or np.random.default_rng(1)
    out = np.zeros(round(duration_s * sr))
    sos = sps.butter(4, [2_000, 8_000], btype="bandpass", fs=sr, output="sos")
    n = round(6 * decay_ms * sr / 1000)
    k = np.arange(n)
    attack = max(1, round(0.0005 * sr))
    env = np.exp(-k / (decay_ms * sr / 1000)) * np.minimum(1.0, (k + 1) / attack)
    for t in onsets_s:
        start = round(t * sr)
        burst = sps.sosfilt(sos, rng.normal(0.0, 1.0, n)) * env
        burst *= peak / (np.max(np.abs(burst)) or 1.0)
        end = min(len(out), start + n)
        out[start:end] += burst[: end - start]
    return out.astype(np.float32), np.asarray(onsets_s, dtype=float)


def speaker_bleed(
    notes: Sequence[tuple[float, float]],
    duration_s: float,
    *,
    peak: float = 0.15,
    decay_s: float = 0.8,
    sr: int = SR,
) -> tuple[np.ndarray, np.ndarray]:
    """The app's own guitar tone leaking from the speakers into the mic.

    ``notes`` are ``(time_s, frequency_hz)``. Each is a plucked harmonic tone with a
    2 ms attack and harmonics rolled off above ~2 kHz, so most energy is below the
    gate's 2..8 kHz band. Returns ``(signal, note_times_s)``.
    """
    out = np.zeros(round(duration_s * sr))
    for t, f0 in notes:
        start = round(t * sr)
        n = min(len(out) - start, round(5 * decay_s * sr))
        if n <= 0:
            continue
        tt = np.arange(n) / sr
        tone = np.zeros(n)
        for h in range(1, 16):
            fh = f0 * h
            if fh > sr / 2:
                break
            tone += np.sin(2 * np.pi * fh * tt) / h * np.exp(-((fh / 2_000.0) ** 2))
        env = np.exp(-tt / decay_s) * np.minimum(1.0, tt / 0.002)
        tone *= env
        out[start : start + n] += tone * (peak / (np.max(np.abs(tone)) or 1.0))
    return out.astype(np.float32), np.asarray([t for t, _ in notes], dtype=float)


def mix(*signals: np.ndarray) -> np.ndarray:
    """Sum signals of possibly different lengths, padded to the longest."""
    n = max(len(s) for s in signals)
    out = np.zeros(n, dtype=np.float32)
    for s in signals:
        out[: len(s)] += s
    return out


def blocks(x: np.ndarray, block_size: int = 64) -> Iterator[np.ndarray]:
    """Yield consecutive blocks as a mic callback would see them; the last is zero-padded."""
    for i in range(0, len(x), block_size):
        b = x[i : i + block_size]
        if len(b) < block_size:
            b = np.pad(b, (0, block_size - len(b)))
        yield b


def write_wav(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    wavfile.write(str(path), sr, np.asarray(x, dtype=np.float32))


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    sr, x = wavfile.read(str(path))
    if x.dtype == np.int16:
        x = x / 32768.0
    return np.asarray(x, dtype=np.float32), sr


def _scale_rms(x: np.ndarray, rms: float) -> np.ndarray:
    cur = float(np.sqrt(np.mean(x**2))) or 1.0
    return (x * (rms / cur)).astype(np.float32)
