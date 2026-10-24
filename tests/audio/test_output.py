"""Output wrapper (with a fake stream), master stage and offline rendering."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from racketguitar.audio.output import (
    CHANNELS,
    QUEUE_MAX,
    AudioOutput,
    Master,
    render_offline,
    render_to_wav,
)
from racketguitar.audio.synth import KarplusStrongSynth
from racketguitar.config import AudioConfig
from tests.audio.measure import SR, all_off, damp, on, rms
from tests.fakes import FakeClock, read_wav


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.latency = 0.0058
        self.started = self.stopped = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeFactory:
    def __init__(self):
        self.streams: list[FakeStream] = []

    def __call__(self, **kwargs):
        s = FakeStream(**kwargs)
        self.streams.append(s)
        return s


def note_at(t: float):
    return on(0, 40, 100, t=t)


def time_info(current: float = 100.0, dac: float = 100.004):
    return SimpleNamespace(currentTime=current, outputBufferDacTime=dac)


def pull(out: AudioOutput, blocks: int, frames: int = 128, ti=None) -> np.ndarray:
    """Run the callback ``blocks`` times; return the (n, channels) float32 output."""
    res = []
    for _ in range(blocks):
        buf = np.full((frames, out.channels), np.nan, dtype=np.float32)
        out.callback(buf, frames, ti if ti is not None else time_info(), None)
        res.append(buf)
    return np.concatenate(res)


# Master stage


def test_master_is_linear_below_the_knee():
    m = Master(0.5, 128)
    x = np.linspace(-1.5, 1.5, 128)
    m.process(x)
    np.testing.assert_allclose(x, np.linspace(-0.75, 0.75, 128))


def test_master_soft_clips_into_unit_range():
    m = Master(1.0, 128)
    x = np.linspace(-50, 50, 128)
    m.process(x)
    assert np.all(np.abs(x) <= 1.0)
    x = np.linspace(-3, 3, 128)
    m.process(x)
    assert np.all(np.abs(x) < 1.0) and np.all(np.diff(x) > 0)  # still monotonic
    y = np.array([0.5, -0.79, 0.9, -2.0, AudioConfig.clip_knee])
    m.process(y)
    assert y[0] == 0.5 and y[1] == -0.79 and y[4] == AudioConfig.clip_knee
    assert AudioConfig.clip_knee < y[2] < 0.9 and -1.0 < y[3] < -AudioConfig.clip_knee


# AudioOutput with a fake stream


def test_start_opens_a_low_latency_stream_and_stop_closes_it():
    f = FakeFactory()
    out = AudioOutput(KarplusStrongSynth(), AudioConfig(device=3), stream_factory=f)
    assert not out.running and out.output_latency_s is None
    with out:
        assert out.running
        (s,) = f.streams
        assert s.started
        kw = s.kwargs
        assert kw["samplerate"] == 48_000 and kw["blocksize"] == 128
        assert kw["channels"] == CHANNELS and kw["dtype"] == "float32"
        assert kw["latency"] == "low" and kw["device"] == 3
        assert kw["callback"] == out.callback
        assert out.output_latency_s == pytest.approx(0.0058)
        out.start()  # idempotent
        assert len(f.streams) == 1
    assert s.stopped and s.closed and not out.running


def test_callback_plays_sent_notes_on_every_channel():
    out = AudioOutput(KarplusStrongSynth(), AudioConfig(), stream_factory=FakeFactory())
    silent = pull(out, 4)
    assert np.all(silent == 0.0)
    out.send(on(0, 40, 127))
    y = pull(out, 40)
    assert np.all(np.isfinite(y)) and np.all(np.abs(y) < 1.0)
    assert rms(y[:, 0]) > 0.01
    np.testing.assert_array_equal(y[:, 0], y[:, 1])
    out.send(all_off())
    pull(out, 20)
    assert np.all(pull(out, 2) == 0.0)


def test_callback_handles_odd_frame_counts():
    out = AudioOutput(KarplusStrongSynth(), AudioConfig(), stream_factory=FakeFactory())
    out.send(on(2, 50))
    y = pull(out, 3, frames=300)  # larger than max_block: rendered in pieces
    assert y.shape == (900, CHANNELS) and not np.any(np.isnan(y))


def test_callback_matches_offline_render():
    """Events sent before a callback == offline events at that block's start sample."""
    cfg = AudioConfig()
    out = AudioOutput(KarplusStrongSynth(), cfg, stream_factory=FakeFactory())
    plan = {0: [on(0, 40, 110)], 10: [on(5, 64, 80)], 50: [damp(0)], 90: [on(0, 43)]}
    got = []
    for b in range(150):
        for e in plan.get(b, []):
            out.send(e)
        got.append(pull(out, 1)[:, 0])
    events = [replace(e, t_detect=b * 128 / SR) for b, es in plan.items() for e in es]
    ref = render_offline(KarplusStrongSynth(), events, 150 * 128 / SR, master_gain=cfg.master_gain)
    np.testing.assert_allclose(np.concatenate(got), ref, rtol=0, atol=1e-6)


def test_detect_to_audio_latency_is_recorded():
    clock = FakeClock(10.0)
    out = AudioOutput(
        KarplusStrongSynth(), AudioConfig(), stream_factory=FakeFactory(), clock=clock
    )
    assert out.last_detect_to_audio_ms is None
    out.send(note_at(9.990))  # detected 10 ms before the callback runs
    out.send(damp(1, t=9.995))  # not a note-on: no latency sample
    pull(out, 1, ti=time_info(100.0, 100.004))  # 4 ms until the DAC
    assert out.last_detect_to_audio_ms == pytest.approx(14.0)
    clock.advance(0.1)
    out.send(note_at(10.095))
    pull(out, 1, ti=SimpleNamespace())  # no DAC time: fall back to stream latency
    assert out.last_detect_to_audio_ms == pytest.approx(5.0)  # not running: 0 ms
    np.testing.assert_allclose(out.detect_to_audio_ms(), [14.0, 5.0])


def test_latency_falls_back_to_stream_latency_while_running():
    clock = FakeClock(10.0)
    out = AudioOutput(
        KarplusStrongSynth(), AudioConfig(), stream_factory=FakeFactory(), clock=clock
    )
    out.start()
    out.send(note_at(9.99))
    pull(out, 1, ti=time_info(0.0, 0.0))
    assert out.last_detect_to_audio_ms == pytest.approx(10.0 + 5.8)
    out.stop()


def test_latency_history_is_a_ring():
    clock = FakeClock(10.0)
    out = AudioOutput(
        KarplusStrongSynth(), AudioConfig(), stream_factory=FakeFactory(), clock=clock
    )
    for i in range(1500):
        out.send(note_at(10.0 - i * 1e-6))
        pull(out, 1, ti=time_info(0.0, 0.001))
    lat = out.detect_to_audio_ms()
    assert len(lat) == 1024
    assert lat[-1] == pytest.approx(1.0 + 1499e-3) and lat[0] == pytest.approx(1.0 + 476e-3)


def test_queue_overflow_drops_oldest_and_counts():
    out = AudioOutput(KarplusStrongSynth(), AudioConfig(), stream_factory=FakeFactory())
    for _ in range(QUEUE_MAX + 5):
        out.send(on(0, 40))
    assert out.dropped_events == 5
    pull(out, 1)
    assert out.dropped_events == 5


def test_xruns_are_counted():
    out = AudioOutput(KarplusStrongSynth(), AudioConfig(), stream_factory=FakeFactory())
    buf = np.zeros((128, CHANNELS), dtype=np.float32)
    out.callback(buf, 128, time_info(), "output underflow")
    out.callback(buf, 128, time_info(), None)
    assert out.xruns == 1 and out.callbacks == 2


def test_sample_rate_mismatch_rejected():
    with pytest.raises(ValueError):
        AudioOutput(KarplusStrongSynth(sample_rate=44_100), AudioConfig())


def test_from_config_builds_the_configured_synth(cfg):
    out = AudioOutput.from_config(cfg, stream_factory=FakeFactory())
    assert isinstance(out.synth, KarplusStrongSynth)


# Offline rendering


def test_offline_events_are_sample_accurate():
    start = 500.0
    for offset in (0, 1, 127, 128, 129, 4321):
        t = start + offset / SR
        x = render_offline(KarplusStrongSynth(), [on(1, 45, 120, t=t)], 0.2, start_time=start)
        first = int(np.flatnonzero(x)[0])
        assert first == offset


def test_offline_ordering_and_early_events():
    # An event before start_time plays at sample 0; ties keep input order (on then damp).
    x = render_offline(KarplusStrongSynth(), [on(0, 40, t=-1.0)], 0.01)
    assert x[0] != 0.0
    y = render_offline(KarplusStrongSynth(), [damp(0, t=0.1), on(0, 40, t=0.1)], 0.2)
    z = render_offline(KarplusStrongSynth(), [on(0, 40, t=0.1), damp(0, t=0.1)], 0.2)
    assert rms(y[-480:]) > 0.0 and np.all(z[-480:] == 0.0)


def test_offline_render_is_deterministic_and_matches_wav(tmp_path):
    strum = [on(s, m, 100, t=0.05 + 0.012 * s) for s, m in enumerate((43, 47, 50, 55, 59, 67))]
    events = [*strum, damp(2, t=0.6), on(5, 67, 127, t=0.7), all_off(t=1.2)]
    synth = KarplusStrongSynth()
    a = render_offline(synth, events, 1.5)
    b = render_offline(synth, events, 1.5)  # same synth again: reset makes it identical
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32 and len(a) == round(1.5 * SR)
    assert np.all(np.isfinite(a)) and np.abs(a).max() < 1.0 and rms(a) > 0.01
    path = tmp_path / "strum.wav"
    c = render_to_wav(path, KarplusStrongSynth(), events, 1.5)
    np.testing.assert_array_equal(c, a)
    x, sr = read_wav(path)
    assert sr == SR
    np.testing.assert_array_equal(x, a)


def test_offline_full_chord_is_clipped_softly():
    events = [on(s, m, 127) for s, m in enumerate((43, 47, 50, 55, 59, 67))]
    x = render_offline(KarplusStrongSynth(), events, 0.5, master_gain=4.0)
    assert np.abs(x).max() <= 1.0
