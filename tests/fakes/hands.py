"""Synthetic MediaPipe-style hands and fingertip tracks.

Hand shapes are built in a local frame where the wrist is at the origin, fingers point
toward -y and the palm size (wrist to middle-finger MCP, landmark 0 -> 9) is 1. Curling
is modelled as rotation toward the camera, so a curled finger appears foreshortened and
folded back over the palm in 2D, which is what a webcam sees.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np

from anystring.types import NUM_LANDMARKS, HandObs, RacketPose

from .racket import uv_to_px

Shape = Literal["open", "pick", "fist"]

WRIST = 0
THUMB_TIP = 4
INDEX_PIP, INDEX_DIP, INDEX_TIP = 6, 7, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12

# Per finger: MCP landmark index, MCP position, splay angle (deg from straight up), lengths.
_FINGERS = (
    (5, (-0.32, -0.95), -8.0, (0.45, 0.27, 0.22)),  # index
    (9, (0.00, -1.00), 0.0, (0.50, 0.30, 0.24)),  # middle
    (13, (0.26, -0.95), 7.0, (0.47, 0.28, 0.22)),  # ring
    (17, (0.48, -0.84), 15.0, (0.37, 0.22, 0.20)),  # pinky
)
_CURLS = {  # (MCP, PIP, DIP) flexion in degrees
    "open": (5.0, 5.0, 3.0),
    "fist": (70.0, 100.0, 60.0),
    "pick_index": (55.0, 70.0, 30.0),
}


def _finger(mcp: tuple[float, float], splay: float, lengths, curls) -> list[np.ndarray]:
    a = math.radians(splay)
    direction = np.array([math.sin(a), -math.cos(a)])
    pts, p, bend = [], np.array(mcp, dtype=float), 0.0
    for length, curl in zip(lengths, curls, strict=True):
        bend += math.radians(curl)
        p = p + direction * length * math.cos(bend)
        pts.append(p)
    return pts


def hand_template(shape: Shape) -> np.ndarray:
    """(21, 2) landmarks in the local hand frame for one of the canonical shapes."""
    lm = np.zeros((NUM_LANDMARKS, 2))
    lm[1] = (-0.28, -0.22)  # thumb CMC
    for i, (mcp_idx, mcp, splay, lengths) in enumerate(_FINGERS):
        if shape == "open":
            curls = _CURLS["open"]
        elif shape == "pick" and i == 0:
            curls = _CURLS["pick_index"]
        else:
            curls = _CURLS["fist"]
        lm[mcp_idx] = mcp
        lm[mcp_idx + 1 : mcp_idx + 4] = _finger(mcp, splay, lengths, curls)

    if shape == "open":
        lm[2:5] = [(-0.55, -0.42), (-0.78, -0.60), (-0.98, -0.76)]
    else:
        # Thumb pad pressed against the side of the curled index's middle phalanx.
        side = 0.5 * (lm[INDEX_PIP] + lm[INDEX_DIP]) + np.array([-0.06, 0.0])
        if shape == "fist":
            side = side + np.array([0.15, 0.10])  # thumb wraps over the fingers instead
        lm[2] = lm[1] + 0.40 * (side - lm[1]) + np.array([-0.10, 0.0])
        lm[3] = lm[1] + 0.72 * (side - lm[1]) + np.array([-0.05, 0.0])
        lm[4] = side
    return lm


def place_hand(
    shape: Shape | np.ndarray,
    at: Sequence[float],
    *,
    anchor: int = INDEX_TIP,
    palm_px: float = 80.0,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Pixel landmarks for a hand whose ``anchor`` landmark sits at ``at``.

    ``angle_deg`` rotates the hand in the image (0 = fingers pointing up). ``shape`` may
    be a name or a template array from ``hand_template``.
    """
    tpl = hand_template(shape) if isinstance(shape, str) else np.asarray(shape, dtype=float)
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = (tpl @ rot.T) * palm_px
    return pts - pts[anchor] + np.asarray(at, dtype=float)


def hand_obs(
    t: float,
    landmarks: np.ndarray,
    *,
    jitter_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> HandObs:
    """Wrap landmarks as a HandObs. Jitter, if any, goes into ``raw`` and ``landmarks``."""
    lm = np.asarray(landmarks, dtype=float)
    if jitter_px:
        rng = rng or np.random.default_rng(0)
        lm = lm + rng.normal(0.0, jitter_px, lm.shape)
    return HandObs(t=t, landmarks=lm.copy(), raw=lm.copy())


def strum_track(
    pose: RacketPose,
    *,
    v_start: float,
    v_end: float,
    u: float = 0.5,
    duration_s: float = 0.12,
    fps: float = 60.0,
    t0: float = 0.0,
    shape: Shape = "pick",
    fingertip: int = INDEX_TIP,
    palm_head_lengths: float = 0.27,
    jitter_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[HandObs]:
    """Frames of a strumming hand whose ``fingertip`` sweeps in a straight line in v.

    Frames are sampled at ``fps`` from ``t0`` to ``t0 + duration_s`` inclusive, and the
    racket is held still at ``pose`` (copy the pose per frame if a test moves it).
    """
    n = max(2, round(duration_s * fps) + 1)
    ts = t0 + np.arange(n) / fps
    vs = np.linspace(v_start, v_end, n)
    hand_angle = math.degrees(math.atan2(pose.u.y, pose.u.x)) + 90.0
    rng = rng or np.random.default_rng(0)
    frames = []
    for t, v in zip(ts, vs, strict=True):
        tip = uv_to_px(pose, u, v)
        lm = place_hand(
            shape,
            tip,
            anchor=fingertip,
            palm_px=palm_head_lengths * pose.head_len,
            angle_deg=hand_angle,
        )
        frames.append(hand_obs(float(t), lm, jitter_px=jitter_px, rng=rng))
    return frames


def fretting_hand(
    pose: RacketPose,
    t: float,
    *,
    handle_t: float,
    handle_u_near: float = -0.15,
    handle_u_far: float = -1.0,
    palm_head_lengths: float = 0.27,
) -> HandObs:
    """A fist gripping the handle with its middle-finger knuckle (landmark 9) at
    ``handle_t`` (0 = head end, 1 = handle end) of the handle span along u."""
    u = handle_u_near + handle_t * (handle_u_far - handle_u_near)
    knuckle = uv_to_px(pose, u, 0.0)
    hand_angle = math.degrees(math.atan2(pose.v.y, pose.v.x)) + 90.0
    lm = place_hand(
        "fist",
        knuckle,
        anchor=MIDDLE_MCP,
        palm_px=palm_head_lengths * pose.head_len,
        angle_deg=hand_angle,
    )
    return hand_obs(t, lm)
