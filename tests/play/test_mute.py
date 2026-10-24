from __future__ import annotations

import numpy as np
import pytest

from racketguitar.geometry.racket import RacketFrame
from racketguitar.play.engine import Engine
from racketguitar.play.mute import PalmMute, is_open_hand
from tests.fakes import ScriptedGate, hand_obs, hand_template, make_pose, place_hand, uv_to_px
from tests.play.tracks import sweep

POSE = make_pose()
FPS = 60.0


def _palm_hand(pose, t, shape="open", uv=(0.5, 0.0), *, jitter_px=0.0, rng=None):
    """A hand with its middle knuckle at ``uv``, fingers along +u (lying on the head)."""
    angle = float(np.degrees(np.arctan2(pose.u.y, pose.u.x))) + 90.0
    lm = place_hand(
        shape, uv_to_px(pose, *uv), anchor=9, palm_px=0.27 * pose.head_len, angle_deg=angle
    )
    return hand_obs(t, lm, jitter_px=jitter_px, rng=rng)


def _run(mute, hands, pose=POSE):
    frame = RacketFrame.from_pose(pose)
    return [mute.update(frame, h) for h in hands]


def test_openness_of_templates():
    assert is_open_hand(hand_template("open"))
    assert not is_open_hand(hand_template("pick"))
    assert not is_open_hand(hand_template("fist"))


@pytest.mark.parametrize("jitter_px", [0.0, 0.5])
def test_resting_open_palm_fires_once_after_hold(cfg, rng, jitter_px):
    mute = PalmMute(cfg.mute)
    ts = np.arange(30) / FPS
    fired = _run(mute, [_palm_hand(POSE, t, jitter_px=jitter_px, rng=rng) for t in ts])
    assert sum(fired) == 1
    t_fire = ts[fired.index(True)]
    hold = cfg.mute.hold_ms / 1000
    assert hold - 1e-9 <= t_fire - ts[0] < hold + 1 / FPS
    assert mute.muted


@pytest.mark.parametrize("shape", ["pick", "fist"])
def test_closed_hand_never_mutes(cfg, shape):
    mute = PalmMute(cfg.mute)
    assert not any(_run(mute, [_palm_hand(POSE, t, shape) for t in np.arange(30) / FPS]))


def test_moving_palm_never_mutes(cfg):
    mute = PalmMute(cfg.mute)
    ts = np.arange(30) / FPS
    speed = cfg.mute.max_speed * 1.5
    hands = [_palm_hand(POSE, t, uv=(0.5, -0.2 + speed * t)) for t in ts]
    assert not any(_run(mute, hands))


def test_slow_drift_below_max_speed_still_mutes(cfg):
    mute = PalmMute(cfg.mute)
    ts = np.arange(30) / FPS
    speed = cfg.mute.max_speed * 0.5
    assert any(_run(mute, [_palm_hand(POSE, t, uv=(0.5, speed * t)) for t in ts]))


@pytest.mark.parametrize("uv", [(1.3, 0.0), (0.5, 0.6), (-0.5, 0.0)])
def test_palm_outside_head_never_mutes(cfg, uv):
    mute = PalmMute(cfg.mute)
    assert not any(_run(mute, [_palm_hand(POSE, t, uv=uv) for t in np.arange(30) / FPS]))


def test_rearms_after_hand_closes_or_leaves(cfg):
    mute = PalmMute(cfg.mute)
    ts = iter(np.arange(200) / FPS)
    rest = [_palm_hand(POSE, next(ts)) for _ in range(15)]
    assert sum(_run(mute, rest)) == 1
    # Keep resting: no second fire.
    assert not any(_run(mute, [_palm_hand(POSE, next(ts)) for _ in range(20)]))
    # Close the hand for one frame, rest again: fires again.
    assert _run(mute, [_palm_hand(POSE, next(ts), "pick")]) == [False]
    assert not mute.muted
    assert sum(_run(mute, [_palm_hand(POSE, next(ts)) for _ in range(15)])) == 1
    # Lose the hand, rest again: fires again.
    assert mute.update(RacketFrame.from_pose(POSE), None) is False
    assert sum(_run(mute, [_palm_hand(POSE, next(ts)) for _ in range(15)])) == 1


def test_engine_mute_damps_ringing_strings(cfg):
    gate = ScriptedGate.always()
    engine = Engine(cfg, gate)
    # Partial down-strum: strings 0..3 ring.
    frames = sweep(POSE, v_start=-0.45, v_end=0.08, duration_s=0.1, fps=FPS)
    events = []
    for f in frames:
        events += engine.process(make_pose(f.t), [f])
    played = sorted({e.string for e in events if e.kind == "on"})
    assert played == [0, 1, 2, 3]
    # The hand leaves view for a few frames, then the open palm is laid on the head.
    for i in range(1, 4):
        assert engine.process(make_pose(frames[-1].t + i / FPS), []) == []
    t0 = frames[-1].t + 4 / FPS
    damps = []
    for i in range(20):
        t = t0 + i / FPS
        evs = engine.process(make_pose(t), [_palm_hand(POSE, t)])
        damps += evs
        if evs:
            assert t - t0 >= cfg.mute.hold_ms / 1000 - 1e-9
    assert [e.kind for e in damps] == ["damp"] * len(played)
    assert [e.string for e in damps] == played
    assert all(e.midi is None for e in damps)
    assert engine.snapshot().muted
    # Lift (close) and rest again with nothing ringing: the mute fires, but no events.
    t1 = t0 + 20 / FPS
    assert engine.process(make_pose(t1), [_palm_hand(POSE, t1, "fist")]) == []
    assert not engine.snapshot().muted
    later = [
        engine.process(make_pose(t1 + i / FPS), [_palm_hand(POSE, t1 + i / FPS)])
        for i in range(1, 20)
    ]
    assert all(evs == [] for evs in later)
    assert engine.snapshot().muted
