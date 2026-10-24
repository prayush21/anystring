"""Measure strum-to-sound latency.

``analyze REC.wav``: the PID's acceptance method. Run ``racketguitar play
--probe-tone`` (every note becomes a short 1 kHz beep) and record, with a second device
next to the racket, yourself tapping or brushing a string a few dozen times. Taps are
brush onsets in 3..8 kHz; the app's answers are beep onsets at 1 kHz. Each beep is
paired with the latest tap 5..300 ms before it; the report gives p50 / p95.

``session DIR``: the app's own detect-to-audio log for a recorded session.

    uv run python -m tools.latency_probe analyze take.wav
    uv run python -m tools.latency_probe session recordings/20261024-150000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

from racketguitar.audio.probe import PROBE_HZ
from racketguitar.capture.mic import detect_onsets
from racketguitar.config import MicConfig

__all__ = ["LatencyReport", "tap_times", "beep_times", "analyze", "main"]

TAP_BAND = (3_000.0, 8_000.0)  # above the beep, so the beep never reads as a tap
BEEP_BAND = (PROBE_HZ * 0.7, PROBE_HZ * 1.3)  # wide and low order: short filter ringing
BEEP_RATIO = 10.0  # beep envelope must exceed the band's noise floor by this (energy)
DOMINANCE = 10.0  # ... and the brush band at the same instant, so brushes never read as beeps
EDGE_FRACTION = 0.2  # a beep starts where its envelope first reaches this of its peak
MERGE_S = 0.020
MIN_LAG_S = 0.005
MAX_LAG_S = 0.300
SAME_SOUND_S = 0.004  # a tap onset this close to a beep onset is the beep's own edge


@dataclass(frozen=True)
class LatencyReport:
    taps: int
    beeps: int
    latencies_ms: np.ndarray

    @property
    def p50(self) -> float | None:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms.size else None

    @property
    def p95(self) -> float | None:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms.size else None

    def summary(self) -> str:
        if not self.latencies_ms.size:
            return f"{self.taps} taps, {self.beeps} beeps, no tap -> beep pairs found"
        return (
            f"{self.latencies_ms.size} pairs ({self.taps} taps, {self.beeps} beeps): "
            f"p50 {self.p50:.1f} ms, p95 {self.p95:.1f} ms, max {self.latencies_ms.max():.1f} ms"
            "  [targets: p50 < 40, p95 < 60]"
        )


def tap_times(x: np.ndarray, sr: int) -> np.ndarray:
    cfg = replace(MicConfig(sample_rate=sr), band_low_hz=TAP_BAND[0], band_high_hz=TAP_BAND[1])
    return np.array([o.t for o in detect_onsets(x, cfg)])


def _envelope(x: np.ndarray, sr: int, band: tuple[float, float]) -> np.ndarray:
    sos = sps.butter(2, band, btype="bandpass", fs=sr, output="sos")
    env = sps.sosfiltfilt(sos, x) ** 2  # zero phase: edges are not delayed
    win = max(1, round(0.001 * sr))  # one beep period
    return np.convolve(env, np.ones(win) / win, mode="same")


def beep_times(x: np.ndarray, sr: int) -> np.ndarray:
    """Start times of the probe beeps, at least one burst apart."""
    beep = _envelope(x, sr, BEEP_BAND)
    brush = _envelope(x, sr, (TAP_BAND[0], min(TAP_BAND[1], 0.45 * sr)))
    floor = max(float(np.median(beep)), 1e-12)
    on = np.r_[False, (beep > BEEP_RATIO * floor) & (beep > DOMINANCE * brush), False]
    starts = np.flatnonzero(on[1:] & ~on[:-1])
    ends = np.flatnonzero(~on[1:] & on[:-1])
    # One beep per run of "on" regions separated by less than MERGE_S (its own ripple).
    merge = round(MERGE_S * sr)
    regions: list[list[int]] = []
    for s, e in zip(starts, ends, strict=True):
        if regions and s - regions[-1][1] < merge:
            regions[-1][1] = e
        else:
            regions.append([s, e])
    out = []
    for s, e in regions:
        seg = beep[s:e]
        out.append((s + int(np.argmax(seg >= EDGE_FRACTION * float(seg.max())))) / sr)
    return np.array(out)


def analyze(x: np.ndarray, sr: int) -> LatencyReport:
    x = np.asarray(x, dtype=np.float64)
    beeps = beep_times(x, sr)
    taps = tap_times(x, sr)
    if beeps.size and taps.size:
        taps = taps[[np.min(np.abs(beeps - t)) > SAME_SOUND_S for t in taps]]
    lat = []
    for b in beeps:
        before = taps[(taps <= b - MIN_LAG_S) & (taps >= b - MAX_LAG_S)]
        if before.size:
            lat.append((b - before.max()) * 1000.0)
    return LatencyReport(int(taps.size), int(beeps.size), np.array(lat))


def _read(path: str) -> tuple[np.ndarray, int]:
    sr, raw = wavfile.read(path)
    x = np.asarray(raw, dtype=np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        x /= float(np.iinfo(raw.dtype).max) + 1.0
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, int(sr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="latency_probe", description="Strum-to-sound latency.")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="tap -> beep latency from a second-device WAV")
    a.add_argument("wav")
    s = sub.add_parser("session", help="the app's detect-to-audio log for a session")
    s.add_argument("session")
    args = p.parse_args(argv)
    if args.command == "analyze":
        x, sr = _read(args.wav)
        print(analyze(x, sr).summary())
        return 0
    meta = json.loads((Path(args.session) / "session.json").read_text())
    stats = meta.get("detect_to_audio_ms")
    if not stats:
        print("no detect-to-audio log in this session (was audio on?)")
        return 1
    print(
        f"detect-to-audio over {stats['n']} notes: p50 {stats['p50']:.1f} ms, "
        f"p95 {stats['p95']:.1f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
