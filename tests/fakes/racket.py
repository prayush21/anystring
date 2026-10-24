"""Synthetic racket poses and racket-frame <-> pixel conversion.

These are an independent oracle for tests: production code must not import them.
"""

from __future__ import annotations

import math

import numpy as np

from racketguitar.config import StringsConfig
from racketguitar.types import RacketPose, Vec2


def make_pose(
    t: float = 0.0,
    *,
    origin: tuple[float, float] = (640.0, 500.0),
    angle_deg: float = -90.0,
    head_len: float = 300.0,
    visible: bool = True,
) -> RacketPose:
    """A pose whose ``u`` points at ``angle_deg`` in image coordinates (0 = +x, -90 = up).

    The default is an upright racket: throat low in the frame, tip above it.
    """
    a = math.radians(angle_deg)
    u = Vec2(math.cos(a), math.sin(a))
    return RacketPose(
        t=t,
        visible=visible,
        origin=Vec2(*map(float, origin)),
        u=u,
        v=Vec2(-u.y, u.x),
        head_len=float(head_len),
    )


def uv_to_px(pose: RacketPose, u: float | np.ndarray, v: float | np.ndarray) -> np.ndarray:
    """Racket-frame coordinates (head lengths) to pixels. Broadcasts; last axis is (x, y)."""
    u = np.asarray(u, dtype=float)[..., None]
    v = np.asarray(v, dtype=float)[..., None]
    return (
        np.asarray(pose.origin) + (u * np.asarray(pose.u) + v * np.asarray(pose.v)) * pose.head_len
    )


def px_to_uv(pose: RacketPose, px: np.ndarray) -> np.ndarray:
    """Pixels (..., 2) to racket-frame (..., 2) as (u, v) in head lengths."""
    d = (np.asarray(px, dtype=float) - np.asarray(pose.origin)) / pose.head_len
    return np.stack([d @ np.asarray(pose.u), d @ np.asarray(pose.v)], axis=-1)


def string_slots_v(cfg: StringsConfig | None = None) -> np.ndarray:
    """v position of each string slot, ordered from ``v_min`` to ``v_max``.

    Slot 0 is string 0 (low E) when ``low_e_side == "neg_v"``; tests that need a fixed
    mapping should set that rather than rely on "auto".
    """
    cfg = cfg or StringsConfig()
    return np.linspace(cfg.v_min, cfg.v_max, cfg.count)


def expected_slots_crossed(
    v_start: float, v_end: float, cfg: StringsConfig | None = None
) -> list[int]:
    """Slots a straight sweep from ``v_start`` to ``v_end`` crosses, in crossing order."""
    slots = string_slots_v(cfg)
    lo, hi = sorted((v_start, v_end))
    hit = [i for i, sv in enumerate(slots) if lo < sv < hi]
    return hit if v_end >= v_start else hit[::-1]
