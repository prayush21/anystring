"""Mic contact onsets: the optional second opinion for the strum gate.

``OnsetDetector`` is pure and headless: feed it 64-sample blocks (or any whole number
of blocks) with the ``time.monotonic()`` time of their first sample and it returns
``Onset``s. ``MicStream`` wraps a ``sounddevice.InputStream`` around it, maps the
stream's ADC time to ``time.monotonic()`` and pushes onsets onto a thread-safe deque
that the integrator drains into the gate before each frame.

Detection function. A string brush is broadband noise with a sub-millisecond attack,
while room noise and the app's own guitar tone (speaker bleed) sit mostly below 2 kHz.
A 64-sample FFT has 750 Hz bins, a 64-point window leaks the loud low end into the
high bins, and its frame grid is the only time resolution it offers. So the detector
band-passes in the time domain instead (Butterworth, ``band_low_hz``..``band_high_hz``,
state carried across blocks) and works on the band's energy envelope:

* ``w``: mean band energy over the last ``WINDOW_BLOCKS`` blocks (4 ms). Averaging
  three blocks gives the noise estimate about 40 degrees of freedom, which is what keeps
  false onsets on stationary noise far below one per minute.
* ``floor``: adaptive noise floor, an EMA of ``w`` with ``floor_time_constant_ms``. It
  is frozen while ``w`` is above threshold, so a brush never raises its own floor, unless
  that lasts over ``MicConfig.max_active_ms``: then it is a new noise level and the EMA resumes.
* onset when ``w > threshold_ratio * floor`` **and** ``w > rise_ratio * r``, where ``r``
  is the peak ``w`` over the ``REF_BLOCKS`` (12 ms) before the window. The second
  test is the envelope flux: it lets a new brush fire while the previous one is still
  ringing above the floor, and it ignores the decaying tail. Onsets closer than
  ``min_interval_ms`` merge into the first.

The onset time is then refined to the sample: the first point where a 0.33 ms
envelope reaches ``ONSET_FRACTION`` of the window's peak, less the filter's delay.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from scipy import signal as sps

from racketguitar.config import Calibration, MicConfig
from racketguitar.types import Onset

__all__ = [
    "OnsetDetector",
    "MicStream",
    "adc_to_monotonic",
    "band_rms",
    "calibrate_mic",
    "detect_onsets",
    "mic_margin",
    "FILTER_ORDER",
    "WINDOW_BLOCKS",
    "REF_BLOCKS",
    "MIN_FLOOR_RMS",
]

FILTER_ORDER = 4  # Butterworth prototype order; the band-pass is twice this
WINDOW_BLOCKS = 3  # detection window, 4 ms at 64 samples / 48 kHz
REF_BLOCKS = 9  # reference span before the window: 12 ms, longer than one period of low E
MIN_FLOOR_RMS = 1e-5  # floor never drops below -100 dBFS, so digital silence is safe
SMOOTH_SAMPLES = 16  # envelope length for sample-accurate onset times
ONSET_FRACTION = 0.1  # onset = envelope reaches this fraction of the window peak
NOISE_MARGIN = 4.0  # ... and at least this many times the floor energy
FILTER_DELAY_SAMPLES = 22  # band-pass + envelope delay at the onset, measured on brushes


class OnsetDetector:
    """High-band envelope-flux onset detector (see module docstring). Not thread-safe."""

    def __init__(self, cfg: MicConfig | None = None) -> None:
        self.cfg = cfg or MicConfig()
        c = self.cfg
        self._sr = float(c.sample_rate)
        self._bs = c.block_size
        nyq = c.sample_rate / 2
        high = min(c.band_high_hz, 0.95 * nyq)
        self._sos = sps.butter(
            FILTER_ORDER, [c.band_low_hz, high], btype="bandpass", fs=c.sample_rate, output="sos"
        )
        self._alpha = 1.0 - math.exp(-self._bs / (c.floor_time_constant_ms * 1e-3 * self._sr))
        self._min_floor = MIN_FLOOR_RMS**2
        self._max_active = math.ceil(c.max_active_ms * 1e-3 * self._sr / self._bs)
        self._rise = c.rise_ratio
        self._min_gap = max(1, math.ceil(c.min_interval_ms * 1e-3 * self._sr / self._bs))
        self._warmup_n = max(WINDOW_BLOCKS + 1, math.ceil(c.warmup_ms * 1e-3 * self._sr / self._bs))
        self._hist_len = (WINDOW_BLOCKS + 1) * self._bs
        self._kernel = np.full(SMOOTH_SAMPLES, 1.0 / SMOOTH_SAMPLES)
        self.reset()

    def reset(self) -> None:
        """Forget all state, as if the stream had just started."""
        self._zi = np.zeros((self._sos.shape[0], 2))
        self._e = [0.0] * WINDOW_BLOCKS  # ring of the latest block energies
        self._pe = 0
        self._w = [0.0] * (WINDOW_BLOCKS + REF_BLOCKS)  # ring of past window energies
        self._pw = 0
        self._floor = self._min_floor
        self._warm = np.zeros(self._warmup_n)
        self._nblocks = 0
        self._since = self._min_gap  # blocks since the last onset
        self._active = 0  # consecutive blocks above threshold
        self._y_hist = np.zeros(self._hist_len)  # filtered samples before this chunk
        self.last_window_energy = 0.0

    @property
    def floor_rms(self) -> float:
        """Current adaptive noise floor as a band RMS (full scale = 1)."""
        return math.sqrt(self._floor)

    @property
    def warmed_up(self) -> bool:
        return self._nblocks >= self._warmup_n

    def process(self, samples: np.ndarray, t: float) -> list[Onset]:
        """Process mono samples whose first sample was captured at ``t``.

        ``len(samples)`` must be a whole number of ``block_size`` blocks; one block per
        call is the real-time case, longer chunks are filtered in one go (tests, files).
        """
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        bs = self._bs
        if x.size % bs:
            raise ValueError(f"expected a multiple of {bs} samples, got {x.size}")
        y, self._zi = sps.sosfilt(self._sos, x, zi=self._zi)
        nb = x.size // bs
        if nb == 1:
            energies = (float(np.dot(y, y)) / bs,)
        else:
            energies = np.einsum("ij,ij->i", y.reshape(nb, bs), y.reshape(nb, bs)) / bs
        out: list[Onset] = []
        e_ring, w_ring, n_w = self._e, self._w, WINDOW_BLOCKS + REF_BLOCKS
        ratio, alpha = self.cfg.threshold_ratio, self._alpha
        for j in range(nb):
            e_ring[self._pe] = float(energies[j])
            self._pe = (self._pe + 1) % WINDOW_BLOCKS
            w = sum(e_ring) / WINDOW_BLOCKS
            # Reference: the peak of the past windows that do not overlap this one. A
            # peak (not a mean) so a tone's beating between harmonics never looks like a
            # rise; whole windows (not blocks) so it is not a single noisy block.
            pw = self._pw
            r = max(w_ring[(pw - lag) % n_w] for lag in range(WINDOW_BLOCKS, n_w + 1))
            w_ring[pw] = w
            self._pw = (pw + 1) % n_w
            self.last_window_energy = w
            self._nblocks += 1
            self._since += 1
            if self._nblocks <= self._warmup_n:
                self._warm[self._nblocks - 1] = w
                if self._nblocks == self._warmup_n:
                    # Median of the second half: robust to the filter's start-up transient.
                    self._floor = max(float(np.median(self._warm[self._warmup_n // 2 :])), 0.0)
                    self._floor = max(self._floor, self._min_floor)
                continue
            thr = ratio * self._floor
            if w > thr and w > self._rise * r and self._since >= self._min_gap:
                self._since = 0
                t_on = self._locate(y, j, t)
                out.append(Onset(t=t_on, strength=w / self._floor))
            self._active = self._active + 1 if w >= thr else 0
            if self._active == 0 or self._active > self._max_active:
                self._floor += alpha * (w - self._floor)
            if self._floor < self._min_floor:
                self._floor = self._min_floor
        # Keep the last few filtered blocks for locating an onset in the next call.
        n = min(y.size, self._hist_len)
        if n < self._hist_len:
            self._y_hist[:-n] = self._y_hist[n:]
        self._y_hist[-n:] = y[-n:]
        return out

    def _locate(self, y: np.ndarray, j: int, t: float) -> float:
        """Sample-accurate time of an onset detected in block ``j`` of this chunk."""
        bs, L = self._bs, self._hist_len
        end = (j + 1) * bs
        if end >= L:
            seg = y[end - L : end]
        else:
            seg = np.concatenate((self._y_hist[end:], y[:end]))
        env = np.convolve(seg * seg, self._kernel)[: seg.size]
        level = max(ONSET_FRACTION * float(env.max()), NOISE_MARGIN * self._floor)
        i = int(np.argmax(env >= level))  # first crossing; argmax of a bool array
        i = max(0, i - FILTER_DELAY_SAMPLES)
        return t + (end - L + i) / self._sr


def adc_to_monotonic(adc_time: float, current_time: float, now: float) -> float:
    """Map a PortAudio stream time to ``time.monotonic()``.

    ``current_time`` is the stream time at callback entry and ``now`` the monotonic time
    read there, so the block's first sample was captured ``current_time - adc_time``
    seconds before ``now``.
    """
    return now - (current_time - adc_time)


class MicStream:
    """Real-time mic input: runs an ``OnsetDetector`` in the PortAudio callback.

    Onsets go onto ``onsets`` (a bounded deque; ``append``/``popleft`` are thread-safe),
    which the integrator empties with ``drain()`` on the vision thread. ``level`` is the
    latest block's band RMS for the overlay meter. ``stream_factory`` defaults to
    ``sounddevice.InputStream`` and is injectable, so tests never open a device.
    """

    def __init__(
        self,
        cfg: MicConfig | None = None,
        *,
        detector: OnsetDetector | None = None,
        stream_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_pending: int = 256,
    ) -> None:
        self.cfg = cfg or MicConfig()
        self.detector = detector or OnsetDetector(self.cfg)
        self.onsets: deque[Onset] = deque(maxlen=max_pending)
        self.level = 0.0
        self.status_errors = 0  # PortAudio overflow/underflow flags seen
        self._factory = stream_factory
        self._clock = clock
        self._stream: Any = None

    def start(self) -> None:
        if self._stream is not None:
            return
        factory = self._factory
        if factory is None:
            import sounddevice  # imported late: tests never load PortAudio

            factory = sounddevice.InputStream
        self.detector.reset()
        self._stream = factory(
            samplerate=self.cfg.sample_rate,
            blocksize=self.cfg.block_size,
            device=self.cfg.device,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        stream.stop()
        stream.close()

    def __enter__(self) -> MicStream:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def drain(self) -> list[Onset]:
        """Remove and return every pending onset, oldest first."""
        out = []
        while True:
            try:
                out.append(self.onsets.popleft())
            except IndexError:
                return out

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        now = self._clock()
        if status:
            self.status_errors += 1
        adc = getattr(time_info, "inputBufferAdcTime", 0.0)
        cur = getattr(time_info, "currentTime", 0.0)
        if adc and cur:
            t = adc_to_monotonic(adc, cur, now)
        else:  # host API gave no timing: assume the block has only just been captured
            t = now - frames / self.cfg.sample_rate
        found = self.detector.process(indata[:, 0], t)
        self.level = math.sqrt(self.detector.last_window_energy)
        self.onsets.extend(found)


def band_rms(x: np.ndarray, cfg: MicConfig | None = None) -> float:
    """Band RMS of a whole buffer, as the detector sees it (filter start-up skipped)."""
    det = OnsetDetector(cfg)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = sps.sosfilt(det._sos, x)
    y = y[min(len(y) // 2, round(0.01 * det.cfg.sample_rate)) :]
    return float(np.sqrt(np.mean(y * y))) if y.size else 0.0


def calibrate_mic(
    noise: np.ndarray,
    brushes: np.ndarray,
    cfg: MicConfig | None = None,
    base: Calibration | None = None,
) -> Calibration:
    """Measure room noise and brush level for ``Calibration``.

    ``noise`` is a recording of the room with nobody playing, ``brushes`` one of the
    player brushing the strings a few times. ``mic_noise_floor`` is the band RMS of the
    noise; ``mic_brush_level`` the median band RMS of the detector window at each brush
    onset (None if no brush was detected). Both are full-scale RMS in the gate's band.
    """
    cfg = cfg or MicConfig()
    floor = band_rms(noise, cfg)
    det = OnsetDetector(cfg)
    x = np.asarray(brushes, dtype=np.float64).reshape(-1)
    x = x[: x.size - x.size % cfg.block_size]
    levels: list[float] = []
    follow = 0  # blocks left in which to track the current brush's peak
    for i in range(0, x.size, cfg.block_size):
        if det.process(x[i : i + cfg.block_size], i / cfg.sample_rate):
            levels.append(0.0)
            follow = 2 * WINDOW_BLOCKS
        if follow:
            levels[-1] = max(levels[-1], det.last_window_energy)
            follow -= 1
    brush = float(np.sqrt(np.median(levels))) if levels else None
    return replace(base or Calibration(), mic_noise_floor=floor, mic_brush_level=brush)


def mic_margin(cal: Calibration, cfg: MicConfig | None = None) -> float | None:
    """Brush energy over the detection threshold (> 1 means brushes are heard).

    The calibration screen should warn below about 2. None until both levels exist.
    """
    cfg = cfg or MicConfig()
    if not cal.mic_noise_floor or cal.mic_brush_level is None:
        return None
    return (cal.mic_brush_level / cal.mic_noise_floor) ** 2 / cfg.threshold_ratio


def detect_onsets(
    x: Sequence[float] | np.ndarray, cfg: MicConfig | None = None, t0: float = 0.0
) -> list[Onset]:
    """Run a fresh detector over a whole buffer (a file or a recording) starting at ``t0``.

    The tail is zero-padded to a whole block, as a mic callback would deliver it.
    """
    det = OnsetDetector(cfg)
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    pad = (-a.size) % det.cfg.block_size
    return det.process(np.pad(a, (0, pad)) if pad else a, t0)
