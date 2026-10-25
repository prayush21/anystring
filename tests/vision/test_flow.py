"""Markerless tracking on synthetic rackets moving with known transforms."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anystring.config import TrackerConfig
from anystring.vision.flow import FlowTracker
from tests.fakes.scenes import render_racket


def drift(k: int, fps: float = 30.0) -> dict:
    t = k / fps
    return {
        "center_px": (640 + 60 * math.sin(0.9 * t), 360 + 25 * math.cos(1.3 * t)),
        "angle_deg": -90 + 15 * math.sin(0.7 * t),
        "head_len_px": 300 * (1 + 0.08 * math.sin(0.5 * t)),
    }


def run(frames: int, *, occlude: bool = False, fps: float = 30.0):
    first = render_racket(**drift(0))
    tr = FlowTracker(TrackerConfig(), head_half_width=first.head_half_width)
    tr.lock(first.image, first.tip, first.throat, 0.0)
    worst = 0.0
    pose = None
    for k in range(1, frames):
        kw = drift(k, fps)
        occ = None
        if occlude:  # a hand sweeping back and forth across the head
            cx, cy = kw["center_px"]
            occ = (cx + 110 * math.sin(2 * math.pi * k / 20), cy + 20, 55)
        scene = render_racket(**kw, occluder=occ)
        pose = tr.update(scene.image, k / fps)
        assert pose.visible
        tip = np.asarray(pose.origin) + np.asarray(pose.u) * pose.head_len
        worst = max(
            worst,
            float(np.hypot(*(np.asarray(pose.origin) - scene.throat))),
            float(np.hypot(*(tip - scene.tip))),
        )
    return worst, tr, pose


def test_follows_a_moving_racket():
    worst, tr, _ = run(120)
    assert worst < 3.0, f"worst tip/throat error {worst:.2f} px"
    assert len(tr.points) > 40


def test_survives_a_hand_passing_over_the_head():
    worst, _, _ = run(120, occlude=True)
    assert worst < 4.0, f"worst tip/throat error {worst:.2f} px"


def test_racket_removed_leaves_the_pose_in_place():
    """A known limit: optical flow cannot tell the racket left. The static background
    keeps tracking, so the pose stays where it was (it does not drift off); re-lock."""
    first = render_racket()
    tr = FlowTracker(TrackerConfig(), head_half_width=first.head_half_width)
    tr.lock(first.image, first.tip, first.throat, 0.0)
    empty = render_racket(show_racket=False).image
    for k in range(1, 30):
        pose = tr.update(empty, k / 30)
    assert np.hypot(*(np.asarray(pose.origin) - first.throat)) < 2.0


def test_featureless_view_times_out():
    first = render_racket()
    tr = FlowTracker(TrackerConfig(hold_ms=150.0), head_half_width=first.head_half_width)
    tr.lock(first.image, first.tip, first.throat, 0.0)
    blank = np.full_like(first.image, 90)  # covered lens: nothing to track
    seen = [tr.update(blank, k / 30).visible for k in range(1, 10)]
    assert seen[0] and not seen[-1]


def test_unlocked_tracker_reports_no_racket():
    tr = FlowTracker()
    assert not tr.locked
    assert not tr.update(render_racket().image, 1.0).visible


@pytest.mark.parametrize("angle", [-90.0, -30.0, 10.0])
def test_lock_gives_the_clicked_geometry(angle):
    s = render_racket(angle_deg=angle, head_len_px=280)
    tr = FlowTracker(head_half_width=s.head_half_width)
    pose = tr.lock(s.image, s.tip, s.throat, 0.0)
    assert np.allclose(pose.origin, s.throat)
    assert pose.head_len == pytest.approx(280, abs=0.5)
    assert math.degrees(math.atan2(pose.u.y, pose.u.x)) == pytest.approx(angle, abs=0.1)


def sway(k: int, fps: float = 30.0) -> dict:
    """Twenty seconds of a held racket swinging +-25 degrees, head up and to the right."""
    t = k / fps
    return {
        "center_px": (640 + 40 * math.sin(0.6 * t), 330 + 20 * math.cos(0.8 * t)),
        "angle_deg": -60 + 25 * math.sin(0.9 * t),
        "head_len_px": 220.0,
    }


def test_long_sway_does_not_drift():
    """Fitting against the lock-frame anchors (not chaining frame-to-frame fits) keeps a
    long session on the racket; the chained tracker was 17 px off here."""
    first = render_racket(**sway(0))
    tr = FlowTracker(TrackerConfig(), head_half_width=first.head_half_width)
    tr.lock(first.image, first.tip, first.throat, 0.0, handle_end=first.handle_end)
    worst = 0.0
    for k in range(1, 600):
        kw = sway(k)
        cx, cy = kw["center_px"]
        hand = (cx + 80 * math.sin(2 * math.pi * k / 20), cy + 10, 40)
        scene = render_racket(**kw, occluder=hand)
        pose = tr.update(scene.image, k / 30)
        assert pose.visible
        tip = np.asarray(pose.origin) + np.asarray(pose.u) * pose.head_len
        worst = max(
            worst,
            float(np.hypot(*(np.asarray(pose.origin) - scene.throat))),
            float(np.hypot(*(tip - scene.tip))),
        )
    assert worst < 9.0, f"worst tip/throat error {worst:.2f} px"


def test_handle_points_need_the_handle_end():
    s = render_racket(**sway(0))
    without = FlowTracker(head_half_width=s.head_half_width)
    without.lock(s.image, s.tip, s.throat, 0.0)
    with_handle_end = FlowTracker(head_half_width=s.head_half_width)
    with_handle_end.lock(s.image, s.tip, s.throat, 0.0, handle_end=s.handle_end)

    def on_handle(tr: FlowTracker) -> int:
        d = tr.points - s.throat
        along = d @ (s.handle_end - s.throat) / float(np.sum((s.handle_end - s.throat) ** 2))
        return int(np.sum(along > 0.15))  # clear of the head's ring

    assert on_handle(without) == 0
    assert on_handle(with_handle_end) >= 5


@pytest.mark.parametrize("configured", [0.3, 0.55])
def test_ring_follows_the_real_rim_not_the_configured_width(configured):
    s = render_racket()
    tr = FlowTracker(head_half_width=configured)
    tr.lock(s.image, s.tip, s.throat, 0.0)
    # The sprite's rim band runs from the mesh edge (0.42) to 0.47 head lengths.
    assert 0.38 <= tr.rim_half_width <= 0.50


def test_a_racket_swung_out_of_view_is_reported_lost():
    """Features lost in a fast swing are not replaced by whatever is behind the last
    pose, so the tracker reports the racket lost instead of tracking the background."""
    first = render_racket()
    tr = FlowTracker(TrackerConfig(hold_ms=150.0), head_half_width=first.head_half_width)
    tr.lock(first.image, first.tip, first.throat, 0.0)
    seen = []
    for k in range(1, 40):
        x = 640 - 160 * k  # gone off the left edge within a few frames
        gone = x <= -400
        scene = render_racket(show_racket=False) if gone else render_racket(center_px=(x, 360.0))
        seen.append(tr.update(scene.image, k / 30).visible)
    assert not any(seen[10:])
