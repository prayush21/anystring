"""Onset detector on synthetic mic signals, the stream wrapper and mic calibration."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from anystring.capture.mic import (
    MicStream,
    OnsetDetector,
    adc_to_monotonic,
    band_rms,
    calibrate_mic,
    detect_onsets,
    mic_margin,
)
from anystring.config import Calibration, MicConfig
from anystring.types import Onset
from tests.fakes import (
    SR,
    blocks,
    brush_strokes,
    mix,
    read_wav,
    room_noise,
    silence,
    speaker_bleed,
    write_wav,
)

TOL_S = 0.005  # the WP4 done-criterion: onsets within 5 ms of the truth

# Notes across the guitar's range, as the synth would play them.
BLEED_FREQS = (82.4, 110.0, 146.8, 196.0, 246.9, 329.6, 440.0, 659.3, 880.0, 1318.5)


def stroke_times(n: int, seed: int, start: float = 0.3) -> np.ndarray:
    """Brush times 40..300 ms apart: from fast strumming to slow, after the warm-up."""
    rng = np.random.default_rng(seed)
    return start + np.cumsum(rng.uniform(0.04, 0.3, n))


def match(onsets: list[Onset], truth: np.ndarray, tol: float = TOL_S):
    """(errors of the nearest onset to each true time, onsets not within tol of any truth)."""
    ts = np.array([o.t for o in onsets])
    if ts.size == 0:
        return np.full(len(truth), np.inf), []
    errs = np.array([ts[np.argmin(np.abs(ts - t))] - t for t in truth])
    extras = [t for t in ts if np.min(np.abs(truth - t)) > tol] if len(truth) else list(ts)
    return errs, extras


def assert_all_found(onsets, truth):
    errs, extras = match(onsets, truth)
    assert np.max(np.abs(errs)) <= TOL_S, f"max error {np.max(np.abs(errs)) * 1e3:.2f} ms"
    assert not extras, f"extra onsets at {extras}"
    assert len(onsets) == len(truth)
    return np.max(np.abs(errs))


@pytest.mark.parametrize("peak", [0.2, 0.05, 0.01])
def test_brushes_on_silence(peak):
    truth = stroke_times(40, seed=1)
    x, truth = brush_strokes(truth, truth[-1] + 0.3, peak=peak, rng=np.random.default_rng(2))
    on = detect_onsets(x)
    assert_all_found(on, truth)
    assert all(o.strength >= MicConfig().threshold_ratio for o in on)


@pytest.mark.parametrize(
    ("peak", "noise_rms"),
    [(0.2, 0.003), (0.2, 0.01), (0.2, 0.03), (0.05, 0.003), (0.05, 0.01), (0.1, 0.02)],
)
def test_brushes_on_room_noise(peak, noise_rms):
    truth = stroke_times(60, seed=3)
    dur = truth[-1] + 0.3
    b, truth = brush_strokes(truth, dur, peak=peak, rng=np.random.default_rng(4))
    x = mix(b, room_noise(dur, rms=noise_rms, rng=np.random.default_rng(5)))
    assert_all_found(detect_onsets(x), truth)


def test_wav_round_trip(tmp_path):
    truth = stroke_times(30, seed=6)
    dur = truth[-1] + 0.3
    b, truth = brush_strokes(truth, dur, peak=0.1, rng=np.random.default_rng(7))
    x = mix(b, room_noise(dur, rms=0.005, rng=np.random.default_rng(8)))
    path = tmp_path / "brushes.wav"
    write_wav(path, x)
    y, sr = read_wav(path)
    assert sr == SR
    on = detect_onsets(y, t0=100.0)
    assert_all_found(on, truth + 100.0)


def test_streaming_blocks_match_whole_buffer():
    truth = stroke_times(20, seed=9)
    dur = truth[-1] + 0.3
    x = mix(
        brush_strokes(truth, dur, peak=0.05)[0],
        room_noise(dur, rms=0.003, rng=np.random.default_rng(1)),
    )
    det = OnsetDetector()
    streamed = []
    for i, blk in enumerate(blocks(x)):
        streamed += det.process(blk, 50.0 + i * 64 / SR)
    whole = detect_onsets(x, t0=50.0)
    assert len(streamed) == len(whole) == len(truth)
    for a, b in zip(streamed, whole, strict=True):
        assert a.t == pytest.approx(b.t, abs=1e-9)
        assert a.strength == pytest.approx(b.strength)


@pytest.mark.slow
@pytest.mark.parametrize("noise_rms", [0.003, 0.03])
def test_false_onsets_on_long_room_noise(noise_rms):
    minutes = 3.0
    x = room_noise(60 * minutes, rms=noise_rms, rng=np.random.default_rng(11))
    on = detect_onsets(x)
    assert len(on) / minutes < 1.0  # measured: 0 per minute


@pytest.mark.slow
def test_false_onsets_on_long_white_noise():
    # Flat noise puts far more energy in the 2..8 kHz band than room noise does.
    x = np.random.default_rng(12).normal(0.0, 0.01, 120 * SR).astype(np.float32)
    assert len(detect_onsets(x)) / 2.0 < 1.0  # measured: 0


def bleed_notes(n: int, start: float = 0.5, step: float = 0.25):
    return [(start + i * step, BLEED_FREQS[i % len(BLEED_FREQS)]) for i in range(n)]


def attack_lag(t: float, note_t: np.ndarray) -> float:
    """Time from the nearest note attack to ``t`` (negative = just before it)."""
    return float((t - note_t)[np.argmin(np.abs(t - note_t))])


@pytest.mark.parametrize("noise_rms", [0.0, 0.003])
def test_speaker_bleed_alone(noise_rms):
    """The bleed tone itself (2 kHz roll-off, 0.8 s ring) never fires; only some note
    attacks do, and those sit inside the gate's bleed-ignore window."""
    notes = bleed_notes(30)
    x, note_t = speaker_bleed(notes, notes[-1][0] + 1.0)
    if noise_rms:
        x = mix(x, room_noise(len(x) / SR, rms=noise_rms, rng=np.random.default_rng(13)))
    on = detect_onsets(x)
    # Measured: 9 of 30 attacks fire on digital silence, 5 of 30 over room noise.
    assert len(on) <= len(notes) // 2
    for o in on:
        assert -0.002 <= attack_lag(o.t, note_t) <= 0.010


def test_brushes_found_through_speaker_bleed():
    truth = stroke_times(40, seed=14)
    dur = truth[-1] + 0.5
    b, truth = brush_strokes(truth, dur, peak=0.1, rng=np.random.default_rng(15))
    notes = [
        (t, BLEED_FREQS[i % len(BLEED_FREQS)]) for i, t in enumerate(np.arange(0.45, dur, 0.33))
    ]
    bleed, note_t = speaker_bleed(notes, dur)
    x = mix(b, bleed, room_noise(dur, rms=0.003, rng=np.random.default_rng(16)))
    on = detect_onsets(x)
    errs, extras = match(on, truth)
    assert np.max(np.abs(errs)) <= TOL_S
    for t in extras:  # anything else is a note attack, inside the gate's ignore window
        assert -0.002 <= attack_lag(t, note_t) <= 0.010


def test_no_onsets_during_warmup_and_none_from_start_transient():
    # A DC offset and a brush inside the warm-up: the filter's start-up step must not
    # fire, and the warm-up brush is (documented as) not reported.
    x = silence(1.0) + 0.05
    b, _ = brush_strokes([0.05, 0.5], 1.0, peak=0.2)
    on = detect_onsets(x + b)
    assert [round(o.t, 3) for o in on] == [0.5]
    assert MicConfig.warmup_ms / 1000 > 0.05


def test_floor_adapts_to_a_lasting_noise_rise():
    quiet = room_noise(2.0, rms=0.003, rng=np.random.default_rng(17))
    loud = room_noise(4.0, rms=0.03, rng=np.random.default_rng(18))
    x = np.concatenate([quiet, loud])
    truth = np.array([3.5, 4.0, 4.6, 5.2])
    b, truth = brush_strokes(truth, 6.0, peak=0.2, rng=np.random.default_rng(19))
    on = detect_onsets(mix(x, b))
    around_step = [o for o in on if 2.0 <= o.t < 2.5]
    assert len(around_step) <= 1
    after = [o for o in on if o.t >= 3.0]
    assert_all_found(after, truth)


def test_min_interval_merges_close_onsets():
    b, _ = brush_strokes([0.3, 0.31, 0.5], 0.8, peak=0.2, decay_ms=3.0)
    on = detect_onsets(b)
    assert [round(o.t, 2) for o in on] == [0.3, 0.5]


def test_rejects_partial_blocks():
    with pytest.raises(ValueError):
        OnsetDetector().process(np.zeros(100), 0.0)


def test_adc_time_mapping():
    # Stream clock says the block was captured 4 ms before the callback ran.
    assert adc_to_monotonic(10.000, 10.004, now=500.0) == pytest.approx(499.996)


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = self.stopped = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def test_mic_stream_pushes_onsets_with_monotonic_times():
    made = []

    def factory(**kwargs):
        made.append(FakeStream(**kwargs))
        return made[-1]

    now = [0.0]
    cfg = MicConfig()
    mic = MicStream(cfg, stream_factory=factory, clock=lambda: now[0])
    with mic:
        stream = made[0]
        assert stream.started
        assert stream.kwargs["samplerate"] == cfg.sample_rate
        assert stream.kwargs["blocksize"] == cfg.block_size
        assert stream.kwargs["channels"] == 1
        cb = stream.kwargs["callback"]
        truth = stroke_times(5, seed=20)
        x, truth = brush_strokes(truth, truth[-1] + 0.2, peak=0.1)
        stream_t0, mono_t0, latency = 7.0, 1000.0, 0.003  # stream clock vs monotonic
        for i, blk in enumerate(blocks(x, cfg.block_size)):
            adc = stream_t0 + i * cfg.block_size / cfg.sample_rate
            cur = adc + latency
            now[0] = mono_t0 + (cur - stream_t0)
            info = SimpleNamespace(inputBufferAdcTime=adc, currentTime=cur)
            cb(blk.reshape(-1, 1), cfg.block_size, info, None)
        got = mic.drain()
        assert mic.drain() == []
        assert mic.level >= 0.0
    assert stream.stopped and stream.closed
    errs, extras = match(got, truth + mono_t0)
    assert np.max(np.abs(errs)) <= TOL_S and not extras


def test_mic_stream_without_host_timing_uses_callback_time():
    def factory(**kwargs):
        return FakeStream(**kwargs)

    mic = MicStream(stream_factory=factory, clock=lambda: 0.0)
    mic.start()
    cb = mic._stream.kwargs["callback"]
    b, _ = brush_strokes([0.3], 0.5, peak=0.2)
    for i, blk in enumerate(blocks(b)):
        now = (i + 1) * 64 / SR  # callback runs as the block completes
        mic._clock = lambda now=now: now
        cb(blk.reshape(-1, 1), 64, SimpleNamespace(inputBufferAdcTime=0.0, currentTime=0.0), None)
    mic.stop()
    (o,) = mic.drain()
    assert o.t == pytest.approx(0.3, abs=TOL_S)


def test_calibrate_mic_measures_floor_and_brush_level():
    noise = room_noise(3.0, rms=0.003, rng=np.random.default_rng(21))
    truth = stroke_times(10, seed=22)
    dur = truth[-1] + 0.3
    b, _ = brush_strokes(truth, dur, peak=0.1)
    brushes = mix(b, room_noise(dur, rms=0.003, rng=np.random.default_rng(23)))
    base = Calibration(grip_closed_score=0.2, grip_open_score=0.8)
    cal = calibrate_mic(noise, brushes, base=base)
    assert cal.grip_closed_score == 0.2  # other fields kept
    assert cal.mic_noise_floor == pytest.approx(band_rms(noise), rel=1e-6)
    assert cal.mic_brush_level is not None and cal.mic_brush_level > 5 * cal.mic_noise_floor
    assert mic_margin(cal) > 2.0
    assert mic_margin(Calibration()) is None
    # Nothing to hear: no brush level, and the margin stays unknown.
    quiet = calibrate_mic(noise, noise)
    assert quiet.mic_brush_level is None and mic_margin(quiet) is None
