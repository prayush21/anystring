"""Signal measurements for the audio tests: pitch, level and brightness."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from racketguitar.audio.synth import KarplusStrongSynth
from racketguitar.types import NoteEvent

SR = 48_000


def on(string: int, midi: int, velocity: int = 100, t: float = 0.0) -> NoteEvent:
    return NoteEvent(t, "on", string, midi, velocity)


def damp(string: int, t: float = 0.0) -> NoteEvent:
    return NoteEvent(t, "damp", string, None, 1)


def all_off(t: float = 0.0) -> NoteEvent:
    return NoteEvent(t, "all_off", -1, None, 1)


def render_raw(
    synth: KarplusStrongSynth, n: int, events: Sequence[NoteEvent] = (), block: int = 128
) -> np.ndarray:
    """Handle ``events`` now, then render ``n`` raw synth samples (no master stage)."""
    for e in events:
        synth.handle(e)
    out = np.zeros(n)
    buf = np.zeros(block)
    for i in range(0, n, block):
        m = min(block, n - i)
        synth.render(buf[:m])
        out[i : i + m] = buf[:m]
    return out


def cents(f: float, ref: float) -> float:
    return 1200.0 * float(np.log2(f / ref))


def f0_spectral(x: np.ndarray, sr: int, guess: float) -> float:
    """Fundamental by the Hann-windowed spectrum peak within a semitone of ``guess``.

    Zero-padded to 2^21 points and refined by a parabola through the log magnitudes.
    """
    nfft = 1 << 21
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    idx = np.flatnonzero((freqs > guess * 2 ** (-1 / 12)) & (freqs < guess * 2 ** (1 / 12)))
    k = int(idx[np.argmax(mag[idx])])
    assert idx[0] < k < idx[-1], "peak at the edge of the search band"
    a, b, c = np.log(mag[k - 1 : k + 2])
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * sr / nfft


def f0_nsdf(x: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 2000.0) -> float:
    """Fundamental by McLeod's normalised square difference (needs no pitch guess).

    Takes the first NSDF maximum above 0.9 of the global maximum, with parabolic
    interpolation of the lag.
    """
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spec * np.conj(spec))[:n]
    sq = np.concatenate([[0.0], np.cumsum(x**2)])
    lags = np.arange(n)
    m = (sq[n] - sq[lags]) + sq[n - lags]  # sum x[j]^2 + x[j+lag]^2 over the overlap
    nsdf = 2.0 * acf / np.maximum(m, 1e-30)
    lo, hi = int(sr / fmax), int(sr / fmin)
    seg = nsdf[lo:hi]
    peaks = [
        i
        for i in range(1, len(seg) - 1)
        if seg[i] > 0 and seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]
    ]
    best = max(seg[i] for i in peaks)
    i = next(i for i in peaks if seg[i] >= 0.9 * best)
    a, b, c = seg[i - 1 : i + 2]
    lag = lo + i + 0.5 * (a - c) / (a - 2 * b + c)
    return sr / lag


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def db(ratio: float) -> float:
    return 20.0 * float(np.log10(ratio))


def spectral_centroid(x: np.ndarray, sr: int) -> float:
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(np.sum(freqs * mag) / np.sum(mag))
