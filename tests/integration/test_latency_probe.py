"""The probe tone and the second-device latency analysis."""

from __future__ import annotations

import json
import tracemalloc

import numpy as np
import pytest
from scipy.io import wavfile

from anystring.audio.output import render_offline
from anystring.audio.probe import BURST_MS, PROBE_HZ, ProbeTone
from anystring.audio.synth import Synth
from anystring.types import NoteEvent
from tests.fakes import brush_strokes, mix, room_noise
from tools.latency_probe import analyze, beep_times, main

SR = 48_000


def take(seed: int, *, noise: float, gain: float, n: int = 30):
    """A synthetic second-device recording: taps, then the app's beep after a lag."""
    rng = np.random.default_rng(seed)
    taps = np.sort(rng.uniform(0.3, 9.5, n))
    taps = taps[np.r_[True, np.diff(taps) > 0.12]]
    lag = rng.uniform(0.025, 0.07, taps.size)
    brush, _ = brush_strokes(taps, 10.0, peak=0.08, rng=rng)
    events = [NoteEvent(t + d, "on", 0, 60, 100) for t, d in zip(taps, lag, strict=True)]
    beeps = render_offline(ProbeTone(), events, 10.0, master_gain=gain)
    return mix(brush, beeps, room_noise(10.0, rms=noise, rng=rng)), lag * 1000.0


def test_probe_tone_is_a_synth():
    p = ProbeTone()
    assert isinstance(p, Synth)
    out = np.zeros(128)
    p.render(out)
    assert not out.any()
    x = render_offline(ProbeTone(), [NoteEvent(0.1, "on", 0, 60, 100)], 0.3, master_gain=1.0)
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(x.size, 1 / SR)
    assert abs(f[np.argmax(spec)] - PROBE_HZ) < 30
    on = np.flatnonzero(np.abs(x) > 1e-4)
    assert abs(on[0] / SR - 0.1) < 0.001
    assert abs((on[-1] - on[0]) / SR * 1000 - BURST_MS) < 2


def test_probe_render_allocates_nothing():
    p = ProbeTone()
    out = np.zeros(128)
    p.handle(NoteEvent(0.0, "on", 0, 60, 100))
    p.render(out)
    tracemalloc.start()
    for _ in range(100):
        p.render(out)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 2048


@pytest.mark.parametrize("noise, gain", [(0.002, 1.0), (0.003, 0.5), (0.02, 0.3), (0.005, 0.1)])
def test_latency_recovered_from_a_recording(noise, gain):
    x, truth = take(7, noise=noise, gain=gain)
    r = analyze(x, SR)
    assert r.latencies_ms.size == truth.size == r.beeps
    assert abs(r.p50 - np.percentile(truth, 50)) < 2.0
    assert abs(r.p95 - np.percentile(truth, 95)) < 2.0


def test_beep_onsets_are_accurate_at_any_level():
    t_true = np.arange(0.5, 9.5, 0.37)
    for gain in (0.05, 0.3, 1.0):
        events = [NoteEvent(t, "on", 0, 60, 100) for t in t_true]
        x = render_offline(ProbeTone(), events, 10.0, master_gain=gain)
        b = beep_times(np.asarray(mix(x, room_noise(10.0, rms=0.005)), float), SR)
        assert b.size == t_true.size
        assert np.abs(b - t_true).max() < 0.002


def test_brushes_alone_are_not_beeps():
    x, _ = brush_strokes(np.arange(0.2, 4.0, 0.25), 4.5, peak=0.3)
    assert beep_times(np.asarray(mix(x, room_noise(4.5, rms=0.003)), float), SR).size == 0


def test_cli(tmp_path, capsys):
    x, _ = take(3, noise=0.003, gain=0.5)
    wavfile.write(tmp_path / "take.wav", SR, (x * 32767).astype(np.int16))
    assert main(["analyze", str(tmp_path / "take.wav")]) == 0
    assert "p50" in capsys.readouterr().out
    (tmp_path / "s").mkdir()
    meta = {"detect_to_audio_ms": {"p50": 11.0, "p95": 14.5, "n": 40}}
    (tmp_path / "s" / "session.json").write_text(json.dumps(meta))
    assert main(["session", str(tmp_path / "s")]) == 0
    assert "p95 14.5" in capsys.readouterr().out
