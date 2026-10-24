from __future__ import annotations

import pytest

from racketguitar.config import DEFAULT_CHORDS, STANDARD_TUNING, ChordConfig
from racketguitar.play.chords import ChordSelector
from racketguitar.play.engine import Engine
from tests.fakes import ScriptedGate, fretting_hand, make_pose
from tests.play.tracks import sweep


def _selector(cfg):
    return ChordSelector(cfg.chords, cfg.strings)


@pytest.mark.parametrize(
    "t,name", [(0.0, "C"), (0.125, "C"), (0.375, "G"), (0.625, "Am"), (0.875, "F"), (1.0, "F")]
)
def test_zone_centres(cfg, t, name):
    sel = _selector(cfg)
    sel.update(0.0, t)
    assert sel.chord == name
    assert sel.voicing == dict(DEFAULT_CHORDS)[name]


def test_out_of_range_clamps_to_end_sections(cfg):
    sel = _selector(cfg)
    sel.update(0.0, -0.04)
    assert sel.chord == "C"
    sel.update(0.1, 1.08)
    assert sel.chord == "F"


def test_hysteresis_at_every_boundary(cfg):
    h = cfg.chords.hysteresis
    assert h == 0.05
    sel = _selector(cfg)
    t = 0.0
    for z in range(3):
        edge = (z + 1) / 4
        sel.update(t, edge - 0.1)
        assert sel.zone == z
        # Up to the edge plus the hysteresis band: stays.
        for x in (edge - 0.01, edge + 0.01, edge + h - 0.001):
            t += 0.01
            sel.update(t, x)
            assert sel.zone == z, (z, x)
        t += 0.01
        sel.update(t, edge + h + 0.001)
        assert sel.zone == z + 1
        # Coming back: holds the new section until past the band on the other side.
        for x in (edge - 0.01, edge - h + 0.001):
            t += 0.01
            sel.update(t, x)
            assert sel.zone == z + 1, (z, x)
        t += 0.01
        sel.update(t, edge - h - 0.001)
        assert sel.zone == z
        t += 0.01
        sel.update(t, edge + h + 0.001)
        assert sel.zone == z + 1


def test_jump_across_sections(cfg):
    sel = _selector(cfg)
    sel.update(0.0, 0.1)
    sel.update(0.1, 0.9)
    assert sel.chord == "F"


def test_no_fretting_hand_means_open_strings_after_hold(cfg):
    sel = _selector(cfg)
    assert sel.zone is None and sel.chord is None
    assert sel.voicing == STANDARD_TUNING
    sel.update(1.0, 0.4)
    assert sel.chord == "G"
    hold = ChordConfig().fret_loss_hold_ms / 1000
    sel.update(1.0 + hold * 0.9, None)
    assert sel.chord == "G"  # a short dropout keeps the chord
    sel.update(1.0 + hold * 1.1, None)
    assert sel.chord is None
    assert sel.voicing == STANDARD_TUNING
    # Back on the handle near an edge: no stale hysteresis, plain section.
    sel.update(2.0, 0.26)
    assert sel.chord == "G"


def test_rejects_bad_voicing(cfg):
    bad = cfg.with_overrides({"chords": {"chords": [["X", [40, 45]]]}})
    with pytest.raises(ValueError):
        _selector(bad)


def _strum_with_fret(cfg, handle_t, *, v_start=-0.45, v_end=0.45):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    pose0 = make_pose()
    frames = sweep(pose0, v_start=v_start, v_end=v_end, duration_s=0.1, fps=60.0)
    events = []
    for f in frames:
        pose = make_pose(f.t)
        hands = [f] if handle_t is None else [fretting_hand(pose, f.t, handle_t=handle_t), f]
        events += engine.process(pose, hands)
    return engine, events


@pytest.mark.parametrize("handle_t,name", [(0.1, "C"), (0.4, "G"), (0.6, "Am"), (0.9, "F")])
def test_engine_plays_the_section_chord(cfg, handle_t, name):
    engine, events = _strum_with_fret(cfg, handle_t)
    voicing = dict(DEFAULT_CHORDS)[name]
    expected = [(s, m) for s, m in enumerate(voicing) if m is not None]
    assert [(e.string, e.midi) for e in events] == expected
    assert all(e.kind == "on" for e in events)
    assert engine.snapshot().chord == name
    assert engine.snapshot().zone == ("C", "G", "Am", "F").index(name)


def test_unplayed_strings_emit_nothing(cfg):
    _, events = _strum_with_fret(cfg, 0.1)  # C: x32010
    assert 0 not in [e.string for e in events]
    assert len(events) == 5


def test_engine_open_strings_without_fretting_hand(cfg):
    engine, events = _strum_with_fret(cfg, None)
    assert [(e.string, e.midi) for e in events] == list(enumerate(STANDARD_TUNING))
    assert engine.snapshot().chord is None
    assert engine.snapshot().voicing == STANDARD_TUNING


def test_chord_change_takes_effect_on_next_hits(cfg):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    pose0 = make_pose()
    down = sweep(pose0, v_start=-0.45, v_end=0.45, duration_s=0.1, fps=60.0)
    up = sweep(pose0, v_start=0.45, v_end=-0.45, duration_s=0.1, fps=60.0, t0=down[-1].t + 1 / 60)
    events = []
    for f in down:
        pose = make_pose(f.t)
        events += engine.process(pose, [fretting_hand(pose, f.t, handle_t=0.4), f])  # G
    for f in up:
        pose = make_pose(f.t)
        events += engine.process(pose, [f, fretting_hand(pose, f.t, handle_t=0.9)])  # F
    g, f_ = dict(DEFAULT_CHORDS)["G"], dict(DEFAULT_CHORDS)["F"]
    assert [e.midi for e in events[:6]] == list(g)
    assert [e.midi for e in events[6:]] == list(f_)[::-1]
