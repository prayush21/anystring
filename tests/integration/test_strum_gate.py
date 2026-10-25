"""The pick-grip gate through the whole play step: an armed down-strum followed by a
return stroke with the hand relaxed open must play the down-strum only.

On real takes the return hand only relaxes part of the way to a spread hand (50-75 %
of the calibrated closed-to-open span); this scene uses 60 %.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from racketguitar.config import GateConfig
from racketguitar.pipeline import FrameInput, Pipeline
from racketguitar.play.gate import grip_score
from tests.fakes import hand_obs, hand_template, make_pose, place_hand, uv_to_px

FPS = 60.0
REST_V = -0.45  # beyond the low-E side (cfg puts low E on -v)


def relaxed(fraction: float) -> np.ndarray:
    """Hand template whose grip score is ``fraction`` of the way from the default closed
    score to the default open score."""
    g = GateConfig()
    target = g.grip_closed_default + fraction * (g.grip_open_default - g.grip_closed_default)
    p, o = hand_template("pick"), hand_template("open")
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if grip_score(p + mid * (o - p)) < target else (lo, mid)
    return p + 0.5 * (lo + hi) * (o - p)


def performance(open_fraction: float, turn_frames: int):
    """(t, v, shape) per frame: settle, pick-grip down-strum, the hand relaxing at the
    bottom over ``turn_frames`` frames, then the up stroke with the relaxed hand."""
    shapes = [("pick", REST_V)] * 6
    shapes += [("pick", v) for v in np.linspace(REST_V, -REST_V, 8)[1:]]
    for k in range(1, turn_frames + 1):
        shapes.append((relaxed(open_fraction * k / turn_frames), -REST_V))
    shapes += [(relaxed(open_fraction), v) for v in np.linspace(-REST_V, REST_V, 8)[1:]]
    return [(100.0 + i / FPS, v, shape) for i, (shape, v) in enumerate(shapes)]


def play(cfg, open_fraction: float, turn_frames: int):
    pose0 = make_pose(0.0)
    angle = math.degrees(math.atan2(pose0.u.y, pose0.u.x)) + 90.0
    pipe = Pipeline(cfg)
    downs = ups = 0
    for t, v, shape in performance(open_fraction, turn_frames):
        pose = make_pose(t)
        at = uv_to_px(pose, 0.5, v)
        lm = place_hand(shape, at, palm_px=0.27 * pose.head_len, angle_deg=angle)
        pipe.step(FrameInput(t, t, pose, (hand_obs(t, lm),), ()))
    for rec in pipe.snapshot().recent:
        if rec.played:
            downs += rec.hit.direction == "down"
            ups += rec.hit.direction == "up"
    return downs, ups


@pytest.mark.parametrize("turn_frames", [1, 2, 3])
def test_relaxed_return_stroke_is_silent(cfg, turn_frames):
    downs, ups = play(cfg, 0.6, turn_frames)
    assert downs == 6
    assert ups == 0


def test_old_symmetric_gate_played_the_return(cfg):
    """The behaviour this guards against: disarming only past 75 % and after two frames
    let the same relaxed return play as an up-strum."""
    old = cfg.with_overrides({"gate": {"grip_release_above": 0.75, "grip_release_frames": 2}})
    downs, ups = play(old, 0.6, 2)
    assert downs == 6
    assert ups == 6


def test_closed_grip_return_still_plays(cfg):
    downs, ups = play(cfg, 0.0, 2)
    assert (downs, ups) == (6, 6)
