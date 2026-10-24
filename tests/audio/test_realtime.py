"""Real-time budget: render time per 128-sample block, and no allocation in the callback.

Timings use the median (robust to other test runs sharing the machine) and take the
best of a few attempts. The p99 is reported, not asserted.
"""

from __future__ import annotations

import time
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest

from racketguitar.audio.output import CHANNELS, AudioOutput
from racketguitar.audio.synth import KarplusStrongSynth
from racketguitar.config import AudioConfig
from tests.audio.measure import damp, on

BUDGET_S = 0.5e-3
G_CHORD = (43, 47, 50, 55, 59, 67)  # highest default-chord note: 2 chunks per block
HIGH = (72, 76, 79, 81, 83, 84)  # worst case in the tested range: 3 chunks per block


def _strum(synth: KarplusStrongSynth, notes, velocity: int = 127) -> None:
    for s, m in enumerate(notes):
        synth.handle(on(s, m, velocity))


def _time_blocks(notes, blocks: int = 3000) -> np.ndarray:
    synth = KarplusStrongSynth()
    buf = np.zeros(128)
    _strum(synth, notes)
    for _ in range(100):  # warm-up
        synth.render(buf)
    times = np.empty(blocks)
    for i in range(blocks):
        if i % 300 == 0:
            _strum(synth, notes)  # keep all six voices sounding
        t0 = time.perf_counter()
        synth.render(buf)
        times[i] = time.perf_counter() - t0
    assert all(synth.active)
    return times


@pytest.mark.parametrize("notes", [G_CHORD, HIGH], ids=["g_chord", "high_notes"])
def test_render_block_under_half_a_millisecond(notes, capsys):
    best = None
    for _ in range(3):
        t = _time_blocks(notes)
        if best is None or np.median(t) < np.median(best):
            best = t
        if np.median(best) < BUDGET_S / 4:
            break
    med, p99 = np.median(best) * 1e6, np.percentile(best, 99) * 1e6
    with capsys.disabled():
        print(f"\n[WP3] 6 voices {notes}: median {med:.1f} us, p99 {p99:.1f} us per 128 samples")
    assert med < BUDGET_S * 1e6


def test_callback_block_under_half_a_millisecond(capsys):
    out = AudioOutput(KarplusStrongSynth(), AudioConfig(), stream_factory=lambda **kw: None)
    buf = np.zeros((128, CHANNELS), dtype=np.float32)
    ti = SimpleNamespace(currentTime=0.0, outputBufferDacTime=0.004)
    times = np.empty(2000)
    for i in range(len(times)):
        if i % 200 == 0:
            for s, m in enumerate(G_CHORD):
                out.send(on(s, m, 127))
        t0 = time.perf_counter()
        out.callback(buf, 128, ti, None)
        times[i] = time.perf_counter() - t0
    med, p99 = np.median(times[100:]) * 1e6, np.percentile(times[100:], 99) * 1e6
    with capsys.disabled():
        print(f"\n[WP3] full callback (6 events/200 blocks): median {med:.1f} us, p99 {p99:.1f} us")
    assert med < BUDGET_S * 1e6


def _traced_peak(fn, calls: int) -> tuple[int, int]:
    """(peak bytes above baseline, net bytes retained) over ``calls`` calls of ``fn``."""
    tracemalloc.start()
    try:
        for _ in range(50):  # warm-up outside the measurement (deque blocks, caches)
            fn()
        base, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        for _ in range(calls):
            fn()
        cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak - base, cur - base


# Python creates small transient objects (numpy views, ints) that tracemalloc sees, so
# "allocates nothing" is checked as: the traced peak does not grow with the block size
# (any per-block array would grow 32x from 128 to 4096 samples), stays under a few
# hundred bytes per live view, and nothing but a counter is retained between calls.
SMALL, LARGE = 128, 4096
MAX_PEAK = 4096  # bytes; one float64 block of 128 samples alone is 1024
MAX_GROWTH = 256  # bytes, LARGE vs SMALL block
MAX_RETAINED = 128  # bytes; counters that grow past CPython's small-int cache


def _check(make_step) -> None:
    peaks = {}
    for block in (SMALL, LARGE):
        peak, net = _traced_peak(make_step(block), 100)
        assert net <= MAX_RETAINED, f"block {block}: retained {net} bytes"
        peaks[block] = peak
    assert peaks[SMALL] < MAX_PEAK, peaks
    assert peaks[LARGE] - peaks[SMALL] < MAX_GROWTH, peaks


def test_render_allocates_no_buffers():
    def make(block):
        synth = KarplusStrongSynth(max_block=block)
        _strum(synth, G_CHORD)
        buf = np.zeros(block)
        return lambda: synth.render(buf)

    _check(make)


def test_note_events_allocate_no_buffers():
    def make(block):
        synth = KarplusStrongSynth(max_block=block)
        buf = np.zeros(block)
        events = [on(s, m, 100) for s, m in enumerate(G_CHORD)] + [damp(2)]

        def step():
            for e in events:
                synth.handle(e)
            synth.render(buf)

        return step

    _check(make)


def test_callback_allocates_no_buffers():
    def make(block):
        cfg = AudioConfig(block_size=block)
        out = AudioOutput(
            KarplusStrongSynth(max_block=block), cfg, stream_factory=lambda **kw: None
        )
        buf = np.zeros((block, CHANNELS), dtype=np.float32)
        ti = SimpleNamespace(currentTime=0.0, outputBufferDacTime=0.004)
        loud = [on(s, m, 127) for s, m in enumerate(G_CHORD)]
        out.master.gain = 4.0  # exercise the soft clipper path too

        def step():
            for e in loud:
                out.send(e)  # producer side (deque append) is measured too
            out.callback(buf, block, ti, None)

        return step

    _check(make)
