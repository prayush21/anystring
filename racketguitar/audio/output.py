"""Audio output: a sounddevice stream wrapper and a deterministic offline renderer.

``AudioOutput`` owns the real-time path. Any thread calls ``send(event)``; the events
cross to the audio thread through a ``collections.deque`` (``append`` / ``popleft`` are
atomic in CPython, so there is no lock). The stream callback drains the queue, renders
the synth into a preallocated buffer, applies master gain and a soft clipper, and
copies the result to every output channel. It also records each note's
detect-to-audio time: ``time.monotonic()`` at render minus ``NoteEvent.t_detect``, plus
the stream's reported time until the block reaches the DAC.

The stream comes from an injectable factory, so tests drive ``callback`` directly and
never open a device. ``render_offline`` runs the same synth and master stage over a
list of timed events, placing each event at its exact sample.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from racketguitar.audio.synth import Synth, make_synth
from racketguitar.config import AudioConfig, Config
from racketguitar.types import NoteEvent

__all__ = [
    "Master",
    "AudioOutput",
    "StreamFactory",
    "default_stream_factory",
    "render_offline",
    "render_to_wav",
    "CHANNELS",
]

CHANNELS = 2  # MacBook speakers are stereo; the mono synth is copied to both
QUEUE_MAX = 1024  # pending events; beyond this the oldest is dropped (and counted)
LATENCY_HISTORY = 1024  # detect-to-audio samples kept for stats

StreamFactory = Callable[..., Any]


class Master:
    """Master gain then a soft clipper, in place and allocation-free.

    ``|y| = |x|`` below ``knee``; above it ``knee + (1 - knee) * tanh(...)``, which
    joins with a matching slope and never exceeds 1.
    """

    def __init__(self, gain: float, max_block: int, knee: float = AudioConfig.clip_knee) -> None:
        if not 0.0 <= knee < 1.0:
            raise ValueError(f"clip knee must be in [0, 1): {knee}")
        self.gain = float(gain)
        self.knee = float(knee)
        self._a = np.zeros(max_block)
        self._b = np.zeros(max_block)

    def process(self, x: np.ndarray) -> None:
        x *= self.gain
        k = self.knee
        if max(float(x.max()), -float(x.min())) <= k:
            return
        n = x.shape[0]
        a, b = self._a[:n], self._b[:n]
        np.abs(x, out=a)
        np.sign(x, out=b)
        x *= b  # x = |x|
        np.subtract(a, k, out=a)
        np.maximum(a, 0.0, out=a)
        a *= 1.0 / (1.0 - k)
        np.tanh(a, out=a)
        a *= 1.0 - k
        a += k
        np.minimum(x, a, out=a)
        np.multiply(a, b, out=x)


def default_stream_factory(**kwargs: Any) -> Any:
    import sounddevice  # lazy: importing loads PortAudio

    return sounddevice.OutputStream(**kwargs)


class AudioOutput:
    """Plays a ``Synth`` through a low-latency output stream.

    ``send`` may be called from any thread. ``start``/``stop`` open and close the
    stream. ``callback`` is the stream callback and is public so tests can call it.
    """

    def __init__(
        self,
        synth: Synth,
        cfg: AudioConfig,
        *,
        stream_factory: StreamFactory = default_stream_factory,
        clock: Callable[[], float] = time.monotonic,
        channels: int = CHANNELS,
    ) -> None:
        if synth.sample_rate != cfg.sample_rate:
            raise ValueError("synth and config sample rates differ")
        self.synth = synth
        self.cfg = cfg
        self.channels = channels
        self._factory = stream_factory
        self._clock = clock
        self._stream: Any = None
        self._queue: deque[NoteEvent] = deque(maxlen=QUEUE_MAX)
        self._block = np.zeros(synth.max_block)
        self.master = Master(cfg.master_gain, synth.max_block, cfg.clip_knee)  # gain is live
        self._lat = np.zeros(LATENCY_HISTORY)
        self._lat_count = 0
        self.dropped_events = 0
        self.xruns = 0
        self.callbacks = 0

    @classmethod
    def from_config(cls, cfg: Config, synth: Synth | None = None, **kwargs: Any) -> AudioOutput:
        return cls(synth if synth is not None else make_synth(cfg), cfg.audio, **kwargs)

    # Producer side (any thread)

    def send(self, event: NoteEvent) -> None:
        if len(self._queue) == QUEUE_MAX:
            self.dropped_events += 1
        self._queue.append(event)

    # Stream lifecycle

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = self._factory(
            samplerate=self.cfg.sample_rate,
            blocksize=self.cfg.block_size,
            channels=self.channels,
            dtype="float32",
            latency="low",
            device=self.cfg.device,
            callback=self.callback,
        )
        self._stream.start()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    def __enter__(self) -> AudioOutput:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._stream is not None

    @property
    def output_latency_s(self) -> float | None:
        """The stream's reported output latency, or None when not running."""
        if self._stream is None:
            return None
        lat = self._stream.latency
        return float(lat[-1] if isinstance(lat, tuple) else lat)

    # Latency stats (read from any thread; values may be one note stale)

    def detect_to_audio_ms(self) -> np.ndarray:
        """Recent detect-to-audio times in ms, oldest first (at most LATENCY_HISTORY)."""
        n = self._lat_count
        if n <= LATENCY_HISTORY:
            return self._lat[:n] * 1000.0
        i = n % LATENCY_HISTORY
        return np.concatenate([self._lat[i:], self._lat[:i]]) * 1000.0

    @property
    def last_detect_to_audio_ms(self) -> float | None:
        n = self._lat_count
        return None if n == 0 else float(self._lat[(n - 1) % LATENCY_HISTORY] * 1000.0)

    # Audio thread

    def callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """sounddevice callback: drain events, render, master, write all channels."""
        self.callbacks += 1
        if status:
            self.xruns += 1
        dac = self._dac_delay(time_info)
        now = self._clock()
        synth, queue = self.synth, self._queue
        while queue:
            try:
                ev = queue.popleft()
            except IndexError:  # pragma: no cover - only if another consumer raced us
                break
            synth.handle(ev)
            if ev.kind == "on":
                self._lat[self._lat_count % LATENCY_HISTORY] = now - ev.t_detect + dac
                self._lat_count += 1
        pos = 0
        step = synth.max_block
        while pos < frames:
            n = min(step, frames - pos)
            buf = self._block[:n]
            synth.render(buf)
            self.master.process(buf)
            outdata[pos : pos + n] = buf[:, None]
            pos += n

    def _dac_delay(self, time_info: Any) -> float:
        try:
            d = float(time_info.outputBufferDacTime - time_info.currentTime)
        except (AttributeError, TypeError):
            d = 0.0
        if d > 0.0 and math.isfinite(d):
            return d
        s = self._stream
        if s is None:
            return 0.0
        lat = s.latency
        return float(lat[-1] if isinstance(lat, tuple) else lat)


def render_offline(
    synth: Synth,
    events: Iterable[NoteEvent],
    duration_s: float,
    *,
    start_time: float = 0.0,
    master_gain: float = AudioConfig.master_gain,
    clip_knee: float = AudioConfig.clip_knee,
) -> np.ndarray:
    """Render ``duration_s`` seconds to a mono float32 array, deterministically.

    Each event takes effect at sample ``round((t_detect - start_time) * sr)`` (events
    before ``start_time`` at sample 0; ties keep input order). The synth is ``reset``
    first, so the same inputs always give the same output. Blocks are split at event
    positions and never exceed ``synth.max_block``.
    """
    sr = synth.sample_rate
    total = round(duration_s * sr)
    timed = sorted(
        ((max(0, round((e.t_detect - start_time) * sr)), i, e) for i, e in enumerate(events)),
        key=lambda x: (x[0], x[1]),
    )
    synth.reset()
    master = Master(master_gain, synth.max_block, clip_knee)
    out = np.zeros(total, dtype=np.float32)
    buf = np.zeros(synth.max_block)
    k = 0
    pos = 0
    while pos < total:
        while k < len(timed) and timed[k][0] <= pos:
            synth.handle(timed[k][2])
            k += 1
        end = min(total, pos + synth.max_block)
        if k < len(timed):
            end = min(end, timed[k][0])
        b = buf[: end - pos]
        synth.render(b)
        master.process(b)
        out[pos:end] = b
        pos = end
    return out


def render_to_wav(
    path: str | Path,
    synth: Synth,
    events: Iterable[NoteEvent],
    duration_s: float,
    *,
    start_time: float = 0.0,
    master_gain: float = AudioConfig.master_gain,
    clip_knee: float = AudioConfig.clip_knee,
) -> np.ndarray:
    """``render_offline`` and write the result as a 32-bit float mono WAV."""
    x = render_offline(
        synth,
        events,
        duration_s,
        start_time=start_time,
        master_gain=master_gain,
        clip_knee=clip_knee,
    )
    wavfile.write(str(path), synth.sample_rate, x)
    return x
