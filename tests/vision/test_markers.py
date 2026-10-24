from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from racketguitar.config import FilterConfig, MarkerConfig
from racketguitar.types import RacketPose
from racketguitar.vision.markers import (
    MarkerTracker,
    detect_markers,
    make_detector,
    marker_centre,
    pose_from_centres,
)
from tests.fakes import render_marker_scene

MAX_PX = 2.0
MAX_DEG = 1.0

ANGLES = (-90.0, -30.0, 0.0, 75.0, 150.0, 220.0)
HEAD_LENS = (250.0, 350.0, 500.0)  # px; 40 mm markers are ~33..67 px wide
CENTRES = ((640.0, 360.0), (560.0, 330.0), (720.0, 390.0))
TILTS = ((0.0, 0.0), (40.0, 0.0), (0.0, 30.0), (35.0, 25.0), (-30.0, -20.0))
DEGRADE = {"clean": (0.0, 0.0), "blur": (1.2, 0.0), "noise": (0.0, 10.0), "both": (1.5, 8.0)}


def tip_px(pose: RacketPose) -> np.ndarray:
    return np.asarray(pose.origin) + np.asarray(pose.u) * pose.head_len


def errors(got: RacketPose, truth: RacketPose) -> tuple[float, float, float]:
    e_origin = float(np.hypot(*(np.asarray(got.origin) - truth.origin)))
    e_tip = float(np.hypot(*(tip_px(got) - tip_px(truth))))
    cos = float(np.clip(np.dot(got.u, truth.u), -1.0, 1.0))
    return e_origin, e_tip, math.degrees(math.acos(cos))


@pytest.mark.parametrize("degrade", DEGRADE)
@pytest.mark.parametrize("tilt", TILTS, ids=lambda t: f"tilt{t[0]:g}_{t[1]:g}")
def test_pose_recovered_within_2px_and_1deg(tilt, degrade):
    blur, noise = DEGRADE[degrade]
    detector_cfg = MarkerConfig()
    worst = np.zeros(3)
    rng = np.random.default_rng(7)
    for i, (angle, head_len) in enumerate(itertools.product(ANGLES, HEAD_LENS)):
        scene = render_marker_scene(
            center_px=CENTRES[i % len(CENTRES)],
            angle_deg=angle,
            head_len_px=head_len,
            tilt_across_deg=tilt[0],
            tilt_along_deg=tilt[1],
            blur_sigma=blur,
            noise_std=noise,
            rng=rng,
        )
        tracker = MarkerTracker(detector_cfg, smooth=False)
        pose = tracker.update(scene.image, t=12.5)
        assert pose.visible, (angle, head_len)
        assert pose.t == 12.5
        assert pose.v == (-pose.u.y, pose.u.x)
        assert math.hypot(*pose.u) == pytest.approx(1.0)
        worst = np.maximum(worst, errors(pose, scene.pose))
    assert worst[0] < MAX_PX and worst[1] < MAX_PX and worst[2] < MAX_DEG, worst


def test_corners_match_ground_truth():
    scene = render_marker_scene(angle_deg=20.0, head_len_px=400.0, tilt_across_deg=30.0)
    found = detect_markers(make_detector(MarkerConfig()), scene.image)
    assert set(found) == {0, 1}
    for mid, quad in found.items():
        assert np.max(np.hypot(*(quad - scene.corners[mid]).T)) < MAX_PX


def test_smoothed_tracker_on_noisy_static_scene(rng):
    tracker = MarkerTracker(MarkerConfig(), FilterConfig(), smooth=True)
    worst = np.zeros(3)
    for k in range(20):
        scene = render_marker_scene(
            angle_deg=-70.0, head_len_px=300.0, tilt_along_deg=20.0, noise_std=12.0, rng=rng
        )
        pose = tracker.update(scene.image, t=5.0 + k / 60)
        assert pose.visible
        worst = np.maximum(worst, errors(pose, scene.pose))
    assert worst[0] < MAX_PX and worst[1] < MAX_PX and worst[2] < MAX_DEG, worst


def test_tiny_markers_never_give_a_wrong_pose():
    # Far away and blurred: the detector may miss, but must not invent a pose.
    scene = render_marker_scene(head_len_px=120.0, blur_sigma=1.5)
    pose = MarkerTracker(smooth=False).update(scene.image, t=1.0)
    if pose.visible:
        e = errors(pose, scene.pose)
        assert e[0] < MAX_PX and e[1] < MAX_PX and e[2] < MAX_DEG


def test_empty_frame_is_not_visible():
    pose = MarkerTracker().update(np.full((720, 1280, 3), 110, np.uint8), t=3.0)
    assert not pose.visible and pose.t == 3.0


def test_centre_is_diagonal_intersection_not_mean():
    quad = np.array([(0.0, 0.0), (10.0, 0.0), (6.0, 4.0), (0.0, 4.0)])  # trapezoid
    c = marker_centre(quad)
    # Intersection lies on both diagonals.
    for a, b in ((quad[0], quad[2]), (quad[1], quad[3])):
        d, e = b - a, c - a
        assert abs(d[0] * e[1] - d[1] * e[0]) < 1e-9
    assert not np.allclose(c, quad.mean(axis=0))


def test_pose_from_centres_conventions():
    pose = pose_from_centres(np.array([100.0, 500.0]), np.array([100.0, 200.0]), t=2.0)
    assert pose is not None
    assert pose.origin == (100.0, 500.0)
    assert pose.u == pytest.approx((0.0, -1.0))
    assert pose.v == pytest.approx((1.0, 0.0))
    assert pose.head_len == pytest.approx(300.0)
    assert pose_from_centres(np.zeros(2), np.zeros(2), t=0.0) is None


def test_hold_then_drop():
    cfg = MarkerConfig(hold_ms=150.0)
    scene = render_marker_scene(angle_deg=-80.0)
    both = scene.corners
    only_throat = {cfg.throat_id: both[cfg.throat_id]}
    tracker = MarkerTracker(cfg, smooth=False)

    first = tracker.update_corners(both, 10.0)
    assert first.visible
    held = tracker.update_corners(only_throat, 10.1)
    assert held.visible and held.t == 10.1
    assert (held.origin, held.u, held.v, held.head_len) == (
        first.origin,
        first.u,
        first.v,
        first.head_len,
    )
    still = tracker.update_corners({}, 10.14)
    assert still.visible
    gone = tracker.update_corners(only_throat, 10.16)
    assert not gone.visible and gone.t == 10.16
    assert gone.origin == first.origin  # last geometry kept for the overlay
    back = tracker.update_corners(both, 10.5)
    assert back.visible and back.t == 10.5


def test_before_first_detection_not_visible():
    pose = MarkerTracker(smooth=False).update_corners({}, 1.0)
    assert not pose.visible and pose.t == 1.0 and pose.head_len == 0.0


def test_smoothing_reduces_corner_jitter_and_resets_after_gap(rng):
    cfg = MarkerConfig()
    scene = render_marker_scene()
    tracker = MarkerTracker(cfg, FilterConfig(), smooth=True)
    raw_err, out_err = [], []
    for k in range(240):
        noisy = {mid: c + rng.normal(0, 0.8, c.shape) for mid, c in scene.corners.items()}
        pose = tracker.update_corners(noisy, 1.0 + k / 60)
        if k > 60:
            raw = pose_from_centres(
                marker_centre(noisy[cfg.throat_id]), marker_centre(noisy[cfg.tip_id]), 0.0
            )
            raw_err.append(np.hypot(*(np.asarray(raw.origin) - scene.pose.origin)))
            out_err.append(np.hypot(*(np.asarray(pose.origin) - scene.pose.origin)))
    assert np.mean(out_err) < 0.5 * np.mean(raw_err)

    # After a gap longer than hold_ms the filter restarts at the new position.
    moved = {mid: c + (200.0, 0.0) for mid, c in scene.corners.items()}
    pose = tracker.update_corners(moved, 10.0)
    assert pose.origin[0] == pytest.approx(scene.pose.origin.x + 200.0, abs=1e-6)


def test_ignores_other_marker_ids():
    scene = render_marker_scene()
    cfg = MarkerConfig(tip_id=5, throat_id=1)  # tip marker in the scene is id 0
    tracker = MarkerTracker(cfg, smooth=False)
    pose = tracker.update(scene.image, 1.0)
    assert not pose.visible
    assert set(tracker.corners) == {1}


def test_unknown_dictionary_rejected():
    with pytest.raises(ValueError, match="dictionary"):
        make_detector(MarkerConfig(dictionary="DICT_NOPE"))
