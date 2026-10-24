"""A latency-probe "instrument": every note-on plays a short 1 kHz beep.

Used by ``racketguitar play --probe-tone`` with ``tools/latency_probe.py``: the beep is
narrowband, far from the 3..8 kHz string brush, so a second-device recording separates
taps from the app's answers cleanly. It goes through the same queue, callback and
master stage as the guitar synth, so the measured latency is the real one.
"""

from __future__ import annotations

import numpy as np

from racketguitar.config import Config
from racketguitar.types import NoteEvent

__all__ = ["PROBE_HZ", "BURST_MS", "ProbeTone"]

PROBE_HZ = 1_000.0
BURST_MS = 40.0
RAMP_MS = 1.0  # raised-cosine edges, so the burst has no click spreading into the brush band
LEVEL = 0.5


class ProbeTone:
    """``Synth``-compatible: allocation-free render, one burst at a time."""

    def __init__(self, sample_rate: int = 48_000, max_block: int = 128) -> None:
        self._sr = int(sample_rate)
        self._max_block = int(max_block)
        n = round(BURST_MS * self._sr / 1000)
        ramp = max(1, round(RAMP_MS * self._sr / 1000))
        env = np.ones(n)
        edge = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
        env[:ramp], env[-ramp:] = edge, edge[::-1]
        burst = LEVEL * env * np.sin(2 * np.pi * PROBE_HZ * np.arange(n) / self._sr)
        self._burst = np.concatenate([burst, np.zeros(self._max_block)])
        self._n = n
        self._pos = -1  # sample index into the burst, -1 = silent

    @classmethod
    def from_config(cls, cfg: Config) -> ProbeTone:
        return cls(cfg.audio.sample_rate, cfg.audio.block_size)

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def max_block(self) -> int:
        return self._max_block

    def handle(self, event: NoteEvent) -> None:
        if event.kind == "on":
            self._pos = 0
        elif event.kind == "all_off":
            self._pos = -1

    def render(self, out: np.ndarray) -> None:
        n = out.shape[0]
        if self._pos < 0:
            out.fill(0.0)
            return
        out[:] = self._burst[self._pos : self._pos + n]
        self._pos += n
        if self._pos >= self._n:
            self._pos = -1

    def reset(self) -> None:
        self._pos = -1
