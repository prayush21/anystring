"""Karplus-Strong plucked-string synth: one voice per guitar string.

Each voice is a delay-line loop ``y[n] = sum_k h[k] * y[n - M - k]``. The 5-tap loop
kernel ``h`` is the classic two-point average convolved with a 3rd-order Lagrange
fractional delay and scaled by ``ks_decay``. ``M`` and the Lagrange delay are solved
per note so the loop's phase delay at the fundamental equals the period exactly, which
keeps every note within a fraction of a cent of equal temperament.

Real-time design: every buffer is preallocated and every note's kernel is computed at
construction, so ``handle`` and ``render`` allocate no arrays and take no locks. Because
every kernel tap is at least ``M`` samples in the past, a chunk of up to ``M`` output
samples depends only on history and is computed with one vectorised ``matmul`` over a
sliding-window view. Low strings render a 128-sample block in one chunk; the high E at
MIDI 67 (period ~122) needs two. Each voice keeps two history buffers and ping-pongs
between them, so carrying history forward never copies overlapping memory.

``handle`` and ``render`` must be called from one thread (the audio callback); other
threads hand events over through ``audio.output.AudioOutput.send``.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

from racketguitar.config import AudioConfig, Config
from racketguitar.types import NUM_STRINGS, NoteEvent

__all__ = [
    "Synth",
    "KarplusStrongSynth",
    "make_synth",
    "midi_to_hz",
    "loop_design",
    "MIDI_MIN",
    "MIDI_MAX",
]

MIDI_MIN = 24  # C1; lower notes are moved up by octaves (the delay line must fit HISTORY)
MIDI_MAX = 108  # C8; higher notes are moved down by octaves (chunks get too short)
HISTORY = 2048  # samples of loop history per voice; > longest period (C1 ~ 1468) + taps
TAPS = 5  # two-point average (2) convolved with 3rd-order Lagrange (4)
NOISE_LEN = 1 << 16  # excitation noise table length
NOISE_SEED = 0x5EED  # fixed, so renders are deterministic
NOISE_STRIDE = 7919  # excitation offset step per pluck (prime), so plucks differ
NOISE_STRING_OFFSET = 10_007  # per-string start offset, so strings never share noise
SILENCE_RMS = 1e-5  # a voice whose block RMS falls below this (-100 dB) stops


@runtime_checkable
class Synth(Protocol):
    """What ``audio.output`` drives: the Karplus-Strong synth and the FluidSynth backend.

    Both methods are called from the audio thread only. ``render`` overwrites ``out``
    (a 1-D float64 array of at most ``max_block`` samples) with the next mono samples,
    before master gain and clipping. Events handled before a ``render`` call take
    effect at its first sample.
    """

    @property
    def sample_rate(self) -> int: ...

    @property
    def max_block(self) -> int: ...

    def handle(self, event: NoteEvent) -> None: ...

    def render(self, out: np.ndarray) -> None: ...

    def reset(self) -> None:
        """Silence everything and return to the just-constructed state."""
        ...


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _lagrange3(d: float) -> np.ndarray:
    """3rd-order Lagrange fractional-delay taps for a delay of ``d`` samples."""
    return np.array(
        [
            -(d - 1) * (d - 2) * (d - 3) / 6,
            d * (d - 2) * (d - 3) / 2,
            -d * (d - 1) * (d - 3) / 2,
            d * (d - 1) * (d - 2) / 6,
        ]
    )


def loop_design(f0: float, sample_rate: int, decay: float) -> tuple[int, np.ndarray]:
    """Integer delay ``M`` and loop kernel ``h`` (``TAPS`` long) for fundamental ``f0``.

    The loop delay is ``M`` plus the kernel's phase delay at ``f0``. ``M`` is chosen so
    the Lagrange part sits in its well-behaved middle interval (1..2 samples), then the
    fractional delay is refined by Newton steps on the exact phase delay, which absorbs
    the averaging filter's half sample and Lagrange's small phase error.
    """
    period = sample_rate / f0
    m = math.floor(period - 1.5)
    w = 2.0 * math.pi * f0 / sample_rate
    k = np.arange(TAPS)
    d = period - m - 0.5
    for _ in range(6):
        h = np.convolve([0.5, 0.5], _lagrange3(d))
        tau = -np.angle(np.sum(h * np.exp(-1j * w * k))) / w
        d += (period - m) - tau
    return m, decay * np.convolve([0.5, 0.5], _lagrange3(d))


class KarplusStrongSynth:
    """Six Karplus-Strong voices, one per string (0 = low E).

    * ``on`` restarts that string's voice: its loop is refilled with a new excitation,
      so re-striking never stacks energy. Other strings are untouched.
    * ``damp`` fades the string out with a raised cosine (no click); it is silent after
      ``damp_fade_fraction * damp_ms``.
    * ``all_off`` damps every string.

    Velocity sets loudness (linear amplitude ``velocity / 127``) and brightness: the
    excitation is ``lowpassed + b * (white - lowpassed)`` with ``b`` going from
    ``brightness[0]`` at velocity 1 to ``brightness[1]`` at 127.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        max_block: int = 128,
        *,
        decay: float = 0.996,
        brightness: tuple[float, float] = (0.2, 0.9),
        damp_ms: float = 30.0,
        voice_gain: float = AudioConfig.voice_gain,
        dark_cutoff_hz: float = AudioConfig.dark_cutoff_hz,
        damp_fade_fraction: float = AudioConfig.damp_fade_fraction,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1): {decay}")
        self._sr = int(sample_rate)
        self._max_block = int(max_block)
        self._b_lo, self._b_hi = (float(b) for b in brightness)
        self._gain = float(voice_gain)

        # Per-note tables, indexed by MIDI 0..127 (out-of-range notes folded by octaves).
        self._m = np.zeros(128, dtype=np.int64)
        self._hrev = np.zeros((128, TAPS))
        self._note_hz = np.zeros(128)
        for note in range(128):
            folded = note
            while folded < MIDI_MIN:
                folded += 12
            while folded > MIDI_MAX:
                folded -= 12
            f0 = midi_to_hz(folded)
            m, h = loop_design(f0, self._sr, decay)
            self._m[note] = m
            self._hrev[note] = h[::-1]  # matmul over a window reads oldest sample first
            self._note_hz[note] = f0
        self._m_list = [int(m) for m in self._m]

        # Excitation noise: white (uniform +-1) and a dark lowpassed copy at equal RMS,
        # stored as (dark, white - dark) so mixing needs no temporaries.
        rng = np.random.default_rng(NOISE_SEED)
        white = rng.uniform(-1.0, 1.0, NOISE_LEN + HISTORY)
        a = math.exp(-2.0 * math.pi * dark_cutoff_hz / self._sr)
        dark = lfilter([1.0 - a], [1.0, -a], white)
        dark *= np.std(white) / np.std(dark)
        self._dark = dark
        self._diff = white - dark

        # Damp fade: raised cosine 1 -> 0, zero-padded by one block. By default it ends at
        # 2/3 of damp_ms (20 of 30 ms): long enough not to click, silent within damp_ms.
        n_fade = max(1, round(damp_fade_fraction * damp_ms * self._sr / 1000.0))
        self._fade_len = n_fade
        self._fade = np.zeros(n_fade + self._max_block)
        self._fade[:n_fade] = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_fade) / n_fade))

        # Two history buffers per voice (ping-pong) and their sliding-window views.
        self._bufs = np.zeros((2, NUM_STRINGS, HISTORY + self._max_block))
        self._buf = [[self._bufs[p, v] for v in range(NUM_STRINGS)] for p in range(2)]
        self._win = [[sliding_window_view(b, TAPS) for b in row] for row in self._buf]
        self._scratch = np.zeros(self._max_block)
        self.reset()

    @classmethod
    def from_config(cls, cfg: Config) -> KarplusStrongSynth:
        a = cfg.audio
        return cls(
            a.sample_rate,
            a.block_size,
            decay=a.ks_decay,
            brightness=(a.ks_brightness_min, a.ks_brightness_max),
            damp_ms=cfg.mute.damp_ms,
            voice_gain=a.voice_gain,
            dark_cutoff_hz=a.dark_cutoff_hz,
            damp_fade_fraction=a.damp_fade_fraction,
        )

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def max_block(self) -> int:
        return self._max_block

    @property
    def active(self) -> tuple[bool, ...]:
        """Which strings are sounding (including ones fading out after a damp)."""
        return tuple(self._active)

    def note_hz(self, midi: int) -> float:
        """The frequency a note actually plays at (after octave folding)."""
        return float(self._note_hz[midi])

    def reset(self) -> None:
        self._bufs.fill(0.0)
        self._cur = [0] * NUM_STRINGS  # which ping-pong buffer holds each voice's history
        self._active = [False] * NUM_STRINGS
        self._note = [0] * NUM_STRINGS
        self._fade_pos = [-1] * NUM_STRINGS  # -1 = not damping
        self._plucks = [0] * NUM_STRINGS  # per string, so strings stay independent
        self._energy = [0.0] * NUM_STRINGS  # output energy since the last silence check
        self._counted = [0] * NUM_STRINGS  # samples in _energy

    def handle(self, event: NoteEvent) -> None:
        kind = event.kind
        if kind == "on":
            self._pluck(event.string, event.midi, event.velocity)  # type: ignore[arg-type]
        elif kind == "damp":
            self._damp(event.string)
        elif kind == "all_off":
            for s in range(NUM_STRINGS):
                self._damp(s)

    def _pluck(self, s: int, midi: int, velocity: int) -> None:
        m = self._m_list[midi]
        span = m + TAPS - 1  # samples of history the loop reads
        dst = self._buf[self._cur[s]][s][HISTORY - span : HISTORY]
        o = (s * NOISE_STRING_OFFSET + self._plucks[s] * NOISE_STRIDE) % NOISE_LEN
        self._plucks[s] += 1
        t = (velocity - 1) / 126.0
        b = self._b_lo + (self._b_hi - self._b_lo) * t
        np.multiply(self._diff[o : o + span], b, out=dst)
        dst += self._dark[o : o + span]
        dst -= dst.mean()  # no DC offset in the loop
        dst *= self._gain * velocity / 127.0
        self._note[s] = midi
        self._active[s] = True
        self._fade_pos[s] = -1
        self._energy[s] = 0.0
        self._counted[s] = 0

    def _damp(self, s: int) -> None:
        if self._active[s] and self._fade_pos[s] < 0:
            self._fade_pos[s] = 0

    def render(self, out: np.ndarray) -> None:
        n = out.shape[0]
        out.fill(0.0)
        if n == 0:
            return
        end = HISTORY + n
        for s in range(NUM_STRINGS):
            if not self._active[s]:
                continue
            p = self._cur[s]
            buf = self._buf[p][s]
            win = self._win[p][s]
            midi = self._note[s]
            m = self._m_list[midi]
            hrev = self._hrev[midi]
            pos = HISTORY
            while pos < end:
                c = min(m, end - pos)
                r = pos - m - (TAPS - 1)
                np.matmul(win[r : r + c], hrev, out=buf[pos : pos + c])
                pos += c
            y = buf[HISTORY:end]
            fp = self._fade_pos[s]
            if fp >= 0:
                tmp = self._scratch[:n]
                np.multiply(y, self._fade[fp : fp + n], out=tmp)
                out += tmp
                self._fade_pos[s] = fp + n
                stop = fp + n >= self._fade_len
            else:
                out += y
                # Silence is judged over at least one full block, so the short blocks
                # of an offline render split at events cannot end a note early.
                stop = False
                self._energy[s] += float(np.dot(y, y))
                self._counted[s] += n
                if self._counted[s] >= self._max_block:
                    stop = self._energy[s] < self._counted[s] * SILENCE_RMS**2
                    self._energy[s] = 0.0
                    self._counted[s] = 0
            if stop:
                self._active[s] = False
                self._fade_pos[s] = -1
                continue
            # Carry the newest history into the other buffer and switch to it.
            span = m + TAPS - 1
            q = 1 - p
            self._buf[q][s][HISTORY - span : HISTORY] = buf[end - span : end]
            self._cur[s] = q


def make_synth(cfg: Config) -> Synth:
    """The synth ``cfg.audio.backend`` selects."""
    if cfg.audio.backend == "fluid":
        from racketguitar.audio.fluid import FluidSynthBackend

        return FluidSynthBackend.from_config(cfg)
    return KarplusStrongSynth.from_config(cfg)
