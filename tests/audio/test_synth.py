"""Karplus-Strong synth: pitch, restart, damping, velocity, bounds."""

from __future__ import annotations

import numpy as np
import pytest

from anystring.audio.synth import (
    MIDI_MAX,
    MIDI_MIN,
    KarplusStrongSynth,
    Synth,
    make_synth,
    midi_to_hz,
)
from anystring.config import STANDARD_TUNING
from tests.audio.measure import (
    SR,
    all_off,
    cents,
    damp,
    db,
    f0_nsdf,
    f0_spectral,
    on,
    render_raw,
    rms,
    spectral_centroid,
)

PITCH_NOTES = list(range(36, 85))
MAX_CENTS = 5.0


@pytest.fixture(scope="module")
def synth() -> KarplusStrongSynth:
    return KarplusStrongSynth()


@pytest.fixture(scope="module")
def pitch_errors(synth) -> dict[int, tuple[float, float]]:
    """Cents error per note by two independent estimators (spectral peak, NSDF)."""
    errs = {}
    for midi in PITCH_NOTES:
        synth.reset()
        x = render_raw(synth, SR, [on(0, midi, 100)])
        target = midi_to_hz(midi)
        steady = x[SR // 20 :]  # skip the first 50 ms of pluck transient
        spec = cents(f0_spectral(steady, SR, target), target)
        nsdf = cents(f0_nsdf(steady[: SR // 4], SR), target)
        errs[midi] = (spec, nsdf)
    return errs


def test_estimators_resolve_a_few_cents():
    """The pitch measurement itself can see a 3-cent detune on a decaying tone."""
    t = np.arange(SR) / SR
    for midi in (36, 60, 84):
        f = midi_to_hz(midi) * 2 ** (3 / 1200)
        x = np.exp(-3 * t) * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))
        assert cents(f0_spectral(x, SR, midi_to_hz(midi)), midi_to_hz(midi)) == pytest.approx(
            3.0, abs=0.05
        )
        assert cents(f0_nsdf(x[: SR // 4], SR), midi_to_hz(midi)) == pytest.approx(3.0, abs=0.5)


@pytest.mark.parametrize("midi", PITCH_NOTES)
def test_pitch_within_5_cents(pitch_errors, midi):
    spec, nsdf = pitch_errors[midi]
    assert abs(spec) < MAX_CENTS, f"MIDI {midi}: spectral {spec:+.3f} cents"
    assert abs(nsdf) < MAX_CENTS, f"MIDI {midi}: NSDF {nsdf:+.3f} cents"


def test_pitch_table_report(pitch_errors, capsys):
    worst_spec = max(abs(s) for s, _ in pitch_errors.values())
    worst_nsdf = max(abs(n) for _, n in pitch_errors.values())
    with capsys.disabled():
        print(
            f"\n[WP3] pitch MIDI {PITCH_NOTES[0]}..{PITCH_NOTES[-1]}: max |error| "
            f"spectral {worst_spec:.4f} cents, NSDF {worst_nsdf:.4f} cents"
        )
    assert worst_spec < MAX_CENTS and worst_nsdf < MAX_CENTS


def test_every_string_of_standard_tuning_in_tune(synth):
    for s, midi in enumerate(STANDARD_TUNING):
        synth.reset()
        x = render_raw(synth, SR, [on(s, midi, 90)])[SR // 20 :]
        assert abs(cents(f0_spectral(x, SR, midi_to_hz(midi)), midi_to_hz(midi))) < 1.0


def test_out_of_range_notes_fold_by_octaves(synth):
    assert synth.note_hz(MIDI_MIN - 12) == pytest.approx(midi_to_hz(MIDI_MIN))
    assert synth.note_hz(MIDI_MAX + 5) == pytest.approx(midi_to_hz(MIDI_MAX + 5 - 12))
    for midi in (0, 127):
        synth.reset()
        x = render_raw(synth, SR // 4, [on(2, midi, 127)])
        assert np.all(np.isfinite(x)) and np.abs(x).max() > 0.01


def test_restart_resets_the_voice(synth):
    synth.reset()
    x = render_raw(synth, 2 * SR, [on(3, 55, 100)])
    tail = rms(x[-480:])
    y = render_raw(synth, 480, [on(3, 55, 100)])
    assert db(rms(y) / tail) > 15.0  # restrike after 2 s: a big energy jump
    fresh = KarplusStrongSynth()
    first = render_raw(fresh, 480, [on(3, 55, 100)])
    assert rms(y) == pytest.approx(rms(first), rel=0.5)  # same level as a first pluck


def test_quick_restrikes_do_not_stack(synth):
    """20 restrikes 5 ms apart stay at single-pluck level instead of summing."""
    single = KarplusStrongSynth()
    ref = render_raw(single, 240, [on(3, 55, 127)])
    synth.reset()
    chunks = [render_raw(synth, 240, [on(3, 55, 127)]) for _ in range(20)]
    levels = [rms(c) for c in chunks]
    assert max(levels) < 1.5 * rms(ref)
    assert max(np.abs(c).max() for c in chunks) < 2.0 * np.abs(ref).max()


def test_strings_are_independent():
    """Restarting one string leaves the others untouched (no voice stealing)."""
    a, b, c = KarplusStrongSynth(), KarplusStrongSynth(), KarplusStrongSynth()
    n = SR // 2
    both = np.concatenate([render_raw(a, n, [on(0, 40), on(1, 45)]), render_raw(a, n, [on(1, 47)])])
    only0 = np.concatenate([render_raw(b, n, [on(0, 40)]), render_raw(b, n)])
    only1 = np.concatenate([render_raw(c, n, [on(1, 45)]), render_raw(c, n, [on(1, 47)])])
    np.testing.assert_allclose(both, only0 + only1, atol=1e-12)
    assert a.active[0] and a.active[1]


@pytest.mark.parametrize("midi", [40, 55, 67, 84])
def test_damp_drops_40_db_within_30_ms(synth, midi):
    synth.reset()
    before = render_raw(synth, SR // 2, [on(4, midi, 127)])
    after = render_raw(synth, SR // 10, [damp(4)])
    ref = rms(before[-240:])  # last 5 ms before the damp
    window = after[round(0.025 * SR) : round(0.030 * SR)]  # 25..30 ms after it
    assert db(max(rms(window), 1e-12) / ref) < -40.0
    assert np.all(after[round(0.020 * SR) :] == 0.0)  # silent from 20 ms, inside 30 ms
    assert not synth.active[4]
    # No click: the fade never steps harder than the ringing string did.
    steps = np.abs(np.diff(np.concatenate([before[-1:], after])))
    assert steps.max() <= 1.05 * np.abs(np.diff(before[-2400:])).max()


def test_damp_only_affects_its_string(synth):
    synth.reset()
    render_raw(synth, 4800, [on(0, 40), on(5, 64)])
    render_raw(synth, 4800, [damp(5)])
    assert synth.active == (True, False, False, False, False, False)


def test_damp_on_silent_string_is_harmless(synth):
    synth.reset()
    x = render_raw(synth, 1280, [damp(2), all_off()])
    assert np.all(x == 0.0) and not any(synth.active)


def test_all_off_silences_every_string(synth):
    synth.reset()
    chord = [on(s, m, 120) for s, m in enumerate((43, 47, 50, 55, 59, 67))]
    render_raw(synth, SR // 4, chord)
    after = render_raw(synth, SR // 10, [all_off()])
    assert np.all(after[round(0.030 * SR) :] == 0.0)
    assert not any(synth.active)
    # The next pluck after all_off plays normally.
    assert rms(render_raw(synth, 4800, [on(2, 50, 100)])) > 0.01


def test_restrike_during_damp_cancels_the_fade(synth):
    synth.reset()
    render_raw(synth, 4800, [on(1, 45)])
    render_raw(synth, 480, [damp(1)])
    x = render_raw(synth, SR // 5, [on(1, 45)])
    assert synth.active[1] and rms(x[-480:]) > 0.01


def test_velocity_louder_and_brighter():
    soft, hard = KarplusStrongSynth(), KarplusStrongSynth()
    n = SR // 10
    xs = render_raw(soft, n, [on(2, 50, 30)])
    xh = render_raw(hard, n, [on(2, 50, 127)])
    assert db(rms(xh) / rms(xs)) > 6.0
    cs, ch = spectral_centroid(xs, SR), spectral_centroid(xh, SR)
    assert ch > 1.3 * cs, (cs, ch)
    # Monotonic over the velocity range.
    levels, centroids = [], []
    for v in (1, 40, 80, 127):
        s = KarplusStrongSynth()
        x = render_raw(s, n, [on(2, 50, v)])
        levels.append(rms(x))
        centroids.append(spectral_centroid(x, SR))
    assert levels == sorted(levels) and centroids == sorted(centroids)


def test_output_finite_and_bounded(synth, rng):
    """Random play for 5 s: no NaN/inf, and the raw mix stays well inside [-1, 1]."""
    synth.reset()
    peak = 0.0
    for _ in range(200):
        events = []
        for _ in range(rng.integers(0, 4)):
            kind = rng.choice(["on", "on", "on", "damp", "all_off"])
            s = int(rng.integers(0, 6))
            if kind == "on":
                events.append(on(s, int(rng.integers(36, 85)), int(rng.integers(1, 128))))
            elif kind == "damp":
                events.append(damp(s))
            else:
                events.append(all_off())
        x = render_raw(synth, int(rng.integers(1, 9)) * 128, events)
        assert np.all(np.isfinite(x))
        peak = max(peak, float(np.abs(x).max()))
    assert peak < 1.0


def test_ringing_voice_eventually_stops(synth):
    synth.reset()
    render_raw(synth, 30 * SR, [on(5, 84, 127)])
    assert not synth.active[5]


def test_partial_blocks_match_full_blocks():
    a, b = KarplusStrongSynth(), KarplusStrongSynth()
    full = render_raw(a, 12_800, [on(0, 40), on(5, 67)])
    for e in (on(0, 40), on(5, 67)):
        b.handle(e)
    parts, buf, n = [], np.zeros(128), 0
    for size in [1, 127, 64, 3, 128, 7] * 40:
        if n >= 12_800:
            break
        size = min(size, 12_800 - n)
        b.render(buf[:size])
        parts.append(buf[:size].copy())
        n += size
    # Chunk boundaries change summation order only: equal to rounding.
    np.testing.assert_allclose(np.concatenate(parts), full, rtol=0, atol=1e-12)


def test_tiny_blocks_do_not_end_a_note():
    s = KarplusStrongSynth()
    s.handle(on(0, 40, 1))
    buf = np.zeros(1)
    for _ in range(4800):
        s.render(buf)
    s.render(np.zeros(0))
    assert s.active[0]


def test_reset_is_deterministic():
    s = KarplusStrongSynth()
    x1 = render_raw(s, 9600, [on(0, 40), on(3, 55, 90)])
    s.reset()
    x2 = render_raw(s, 9600, [on(0, 40), on(3, 55, 90)])
    np.testing.assert_array_equal(x1, x2)


def test_from_config_and_protocol(cfg):
    cfg = cfg.with_overrides({"audio": {"ks_decay": 0.99}, "mute": {"damp_ms": 10.0}})
    s = make_synth(cfg)
    assert isinstance(s, KarplusStrongSynth) and isinstance(s, Synth)
    assert s.sample_rate == cfg.audio.sample_rate and s.max_block == cfg.audio.block_size
    render_raw(s, 4800, [on(0, 40)])
    after = render_raw(s, 960, [damp(0)])
    assert np.all(after[480:] == 0.0)  # silent within the 10 ms damp


def test_decay_controls_sustain():
    long, short = KarplusStrongSynth(decay=0.998), KarplusStrongSynth(decay=0.98)
    a = render_raw(long, SR, [on(0, 45)])
    b = render_raw(short, SR, [on(0, 45)])
    assert rms(a[-4800:]) > 5 * rms(b[-4800:])
    with pytest.raises(ValueError):
        KarplusStrongSynth(decay=1.0)
