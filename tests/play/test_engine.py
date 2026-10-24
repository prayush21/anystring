from __future__ import annotations

import numpy as np
import pytest

from racketguitar.config import STANDARD_TUNING, Config, VelocityConfig
from racketguitar.play.engine import Engine, velocity_for_speed
from racketguitar.types import StrumGate
from tests.fakes import (
    ScriptedGate,
    fretting_hand,
    make_pose,
    string_slots_v,
    uv_to_px,
)
from tests.play.tracks import single_tip, sweep, tip_path

POSE = make_pose()


def _play(engine, frames, *, fret_t=None, visible=True, **pose_kw):
    events = []
    for f in frames:
        pose = make_pose(f.t, visible=visible, **pose_kw)
        hands = [f] if fret_t is None else [fretting_hand(pose, f.t, handle_t=fret_t), f]
        events += engine.process(pose, hands)
    return events


def _full_sweep(**kw):
    kw.setdefault("duration_s", 0.1)
    kw.setdefault("fps", 60.0)
    return sweep(POSE, v_start=-0.45, v_end=0.45, **kw)


def test_scripted_gate_satisfies_protocol():
    assert isinstance(ScriptedGate.always(), StrumGate)


def test_gate_never_blocks_everything(cfg):
    gate = ScriptedGate.never()
    engine = Engine(cfg, gate)
    assert _play(engine, _full_sweep()) == []
    assert len(gate.queries) == 6 and not any(ok for _, ok in gate.queries)
    assert gate.emitted == []
    assert [r.played for r in engine.snapshot().recent] == [False] * 6


def test_gate_always_passes(cfg):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    events = _play(engine, _full_sweep())
    assert [e.kind for e in events] == ["on"] * 6
    assert [(e.string, e.midi) for e in events] == list(enumerate(STANDARD_TUNING))
    assert [t for t, _ in gate.queries] == [e.t_detect for e in events]
    assert gate.emitted == [e.t_detect for e in events]
    assert all(cfg.velocity.min_velocity <= e.velocity <= 127 for e in events)


def test_gate_intervals_gate_exactly(cfg):
    """One frame gap holds all six crossings; the gate is open only around strings 2-3
    and string 5, so exactly those play."""
    cfg = single_tip(cfg)
    fps, v0, v1 = 30.0, -0.45, 0.45
    frames = sweep(POSE, v_start=v0, v_end=v1, duration_s=1 / fps, fps=fps)
    t0, t1 = frames[-2].t, frames[-1].t
    th = t0 + (string_slots_v(cfg.strings) - v0) / (v1 - v0) * (t1 - t0)
    mid = (th[:-1] + th[1:]) / 2
    gate = ScriptedGate([(mid[1], mid[3]), (mid[4], t1 + 1)])
    engine = Engine(cfg, gate)
    events = _play(engine, frames)
    assert [e.string for e in events] == [2, 3, 5]
    np.testing.assert_allclose([e.t_detect for e in events], th[[2, 3, 5]], atol=1e-9)
    assert [ok for _, ok in gate.queries] == [False, False, True, True, False, True]
    assert gate.emitted == [e.t_detect for e in events]


def test_observe_hand_once_per_frame_with_strumming_hand(cfg):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    frames = _full_sweep()
    poses, fret = [], []
    for f in frames:
        pose = make_pose(f.t)
        poses.append(pose)
        fret.append(fretting_hand(pose, f.t, handle_t=0.6))
    # Alternate the list order: identification must not depend on it.
    for i, (f, p, fr) in enumerate(zip(frames, poses, fret, strict=True)):
        engine.process(p, [fr, f] if i % 2 else [f, fr])
        assert engine.snapshot().fretting is fr
        assert engine.snapshot().strumming is f
    assert len(gate.hands) == len(frames)
    for (t, hand), f in zip(gate.hands, frames, strict=True):
        assert hand is f
        assert t == f.t


def test_observe_hand_none_without_strumming_hand(cfg):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    t = 0.0
    pose = make_pose(t)
    engine.process(pose, [])  # no hands
    engine.process(
        make_pose(t + 0.02), [fretting_hand(make_pose(t + 0.02), t + 0.02, handle_t=0.3)]
    )
    f = _full_sweep()[0]
    engine.process(make_pose(t + 0.04, visible=False), [f])  # racket not visible
    assert [h for _, h in gate.hands] == [None, None, None]
    assert engine.snapshot().chord == "G"  # the lone hand on the handle is fretting


def test_single_hand_on_head_is_strumming(cfg):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    events = _play(engine, _full_sweep())
    assert len(events) == 6
    assert all(h is not None for _, h in gate.hands)


def test_note_emitted_per_emitted_note_only(cfg):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    events = _play(engine, _full_sweep(), fret_t=0.1)  # C leaves low E unplayed
    assert len(events) == 5
    assert gate.emitted == [e.t_detect for e in events]
    rec = engine.snapshot().recent
    assert [(r.hit.string, r.played, r.midi) for r in rec][0] == (0, False, None)


def test_no_events_while_racket_hidden(cfg):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    assert _play(engine, _full_sweep(), visible=False) == []
    assert gate.queries == []
    assert engine.snapshot().strings_px() is None


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_moving_racket_end_to_end(cfg, fps):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    n = round(0.5 * fps)
    ts = np.arange(n) / fps
    vs = np.r_[[-0.45] * 3, np.linspace(-0.45, 0.45, n - 3)]
    events = []
    poses = [
        make_pose(t, origin=(600 + 200 * t, 480 - 100 * t), angle_deg=-90 + 60 * t, head_len=280)
        for t in ts
    ]
    for pose, f in zip(poses, tip_path(poses, [(0.5, v) for v in vs]), strict=True):
        fr = fretting_hand(pose, pose.t, handle_t=0.9)
        events += engine.process(pose, [f, fr])
    assert [e.string for e in events] == [0, 1, 2, 3, 4, 5]
    assert engine.snapshot().chord == "F"


def test_velocity_curve_monotonic_and_in_range():
    vc = VelocityConfig()
    speeds = np.linspace(0.0, 20.0, 2001)
    vel = np.array([velocity_for_speed(s, vc) for s in speeds])
    assert np.all(np.diff(vel) >= 0)
    assert vel.min() == 20 and vel.max() == 127
    assert velocity_for_speed(vc.speed_at_min, vc) == 20
    assert velocity_for_speed(vc.speed_at_max, vc) == 127
    mid = (vc.speed_at_min + vc.speed_at_max) / 2
    assert velocity_for_speed(mid, vc) == round(20 + 107 * 0.5**1.6)
    inside = vel[(speeds > vc.speed_at_min) & (speeds < vc.speed_at_max)]
    assert len(np.unique(inside)) > 90  # actually graded, not a step


def test_faster_strums_are_louder(cfg):
    vels = []
    for duration in (0.9, 0.3, 0.1, 0.05):
        engine = Engine(cfg, ScriptedGate.always())
        events = _play(engine, _full_sweep(duration_s=duration))
        assert len(events) == 6
        vels.append(events[0].velocity)
    assert vels == sorted(vels) and vels[0] < vels[-1]
    assert all(20 <= v <= 127 for v in vels)


def test_snapshot_geometry_matches_oracle(cfg):
    engine = Engine(cfg, ScriptedGate.always())
    pose = make_pose(0.0, angle_deg=-60.0, head_len=250.0)
    engine.process(pose, [fretting_hand(pose, 0.0, handle_t=0.4)])
    snap = engine.snapshot()
    assert snap.visible and snap.chord == "G" and snap.zone == 1
    assert snap.chord_names == ("C", "G", "Am", "F")
    seg = snap.strings_px()
    for s, v in enumerate(string_slots_v(cfg.strings)):
        np.testing.assert_allclose(seg[s], uv_to_px(pose, [0.15, 0.85], [v, v]))
    sec = snap.handle_sections_px()
    np.testing.assert_allclose(sec[1], uv_to_px(pose, [-0.3625, -0.575], [0.0, 0.0]))
    assert snap.grip in ("closed", "open", "unknown")


def test_auto_orientation_in_engine():
    """Guitar hold with the head to the left (u = -x): v points up, so low E (top of
    the image) is on the pos-v side. A sweep toward +v therefore goes high E -> low E,
    i.e. an up-strum, and string indices count down."""
    cfg = Config()
    assert cfg.strings.low_e_side == "auto"
    engine = Engine(cfg, ScriptedGate.always())
    pose0 = make_pose(angle_deg=180.0)
    frames = sweep(pose0, v_start=-0.45, v_end=0.45, duration_s=0.1, fps=60.0)
    events = []
    for f in frames:
        events += engine.process(make_pose(f.t, angle_deg=180.0), [f])
    assert [e.string for e in events] == [5, 4, 3, 2, 1, 0]
    assert [r.hit.direction for r in engine.snapshot().recent] == ["up"] * 6
    assert not engine.snapshot().low_e_neg_v
    # The top string in the image is string 0.
    seg = engine.snapshot().strings_px()
    assert seg[0, 0, 1] < seg[5, 0, 1]


def test_reset(cfg):
    engine = Engine(cfg, ScriptedGate.always())
    frames = _full_sweep()
    _play(engine, frames[:-2], fret_t=0.4)
    engine.reset()
    assert engine.process(make_pose(frames[-1].t), [frames[-1]]) == []


def test_a_hand_on_the_shaft_is_never_the_strumming_hand():
    """With chords on the grip only, a fist holding the shaft is not fretting; it must
    not be taken for the strumming hand either (it would arm the gate as a fist)."""
    from racketguitar.config import Config
    from racketguitar.geometry.racket import HandleAxis, RacketFrame
    from racketguitar.play.engine import split_hands
    from tests.fakes import hand_obs, make_pose, place_hand, uv_to_px

    cfg = Config().with_overrides({"chords": {"handle_u_near": -1.2, "handle_u_far": -2.0}})
    pose = make_pose()
    frame = RacketFrame.from_pose(pose)
    handle = HandleAxis.from_config(cfg.chords)
    shaft = hand_obs(0.0, place_hand("fist", uv_to_px(pose, -0.5, 0.0), anchor=9, palm_px=80))
    strum = hand_obs(0.0, place_hand("pick", uv_to_px(pose, 0.5, 0.45), palm_px=80))
    far = hand_obs(0.0, place_hand("pick", uv_to_px(pose, 0.5, 2.5), palm_px=80))
    assert split_hands(frame, [shaft], handle, 9) == (None, None)
    assert split_hands(frame, [shaft, strum], handle, 9) == (None, strum)
    assert split_hands(frame, [far], handle, 9) == (None, None)  # nowhere near the head
