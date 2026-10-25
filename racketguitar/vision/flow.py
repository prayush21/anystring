"""Markerless racket tracking: lock on with two clicks, then follow with optical flow.

``lock`` takes the head's tip and the throat (where the head meets the shaft) in
pixels. Corner features are picked on a ring around the head's rim, where a racket has
unique detail (frame graphics, string ends) and the strumming hand rarely is; the
string mesh inside is skipped because its repeating grid fools optical flow. Each
frame the features are followed with pyramidal Lucas-Kanade, checked forward and
backward, and one similarity transform (rotation, uniform scale, shift) is fitted with
RANSAC, so features on a passing hand or the background are outvoted. The transform
moves the tip and throat, which give the ``RacketPose``.

It can drift during fast moves and cannot tell the racket left the frame (the static
background keeps tracking); re-lock to reset. Works at half resolution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from racketguitar.config import TrackerConfig
from racketguitar.types import RacketPose, Vec2

__all__ = ["FlowTracker", "NO_POSE", "pose_from_points"]

# Before a lock, and whenever there is no racket to report.
NO_POSE = RacketPose(
    t=0.0, visible=False, origin=Vec2(0.0, 0.0), u=Vec2(0.0, -1.0), v=Vec2(1.0, 0.0), head_len=0.0
)

SCALE = 0.5  # track on a half-resolution copy
MAX_POINTS = 300
MIN_POINTS = 60  # top up with fresh features below this
MIN_INLIERS = 12
FB_MAX_PX = 1.0  # forward-backward error allowed, tracking pixels
RANSAC_PX = 1.5
MAX_SCALE_STEP = 0.08  # reject a frame whose fitted scale changes more than this
RING = (0.75, 1.15)  # feature ring, as fractions of the head ellipse
_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


class FlowTracker:
    """Not thread-safe; vision thread only."""

    def __init__(self, cfg: TrackerConfig | None = None, *, head_half_width: float = 0.4) -> None:
        self.cfg = cfg or TrackerConfig()
        self.head_half_width = head_half_width
        self.reset()

    def reset(self) -> None:
        self._tip: np.ndarray | None = None  # full-resolution pixels
        self._throat: np.ndarray | None = None
        self._prev: np.ndarray | None = None  # previous small grey frame
        self._pts = np.zeros((0, 1, 2), np.float32)  # small-frame pixels
        self._last_good_t = -math.inf

    @property
    def locked(self) -> bool:
        return self._tip is not None

    @property
    def points(self) -> np.ndarray:
        """Tracked feature points in full-resolution pixels, (n, 2), for the overlay."""
        return self._pts.reshape(-1, 2) / SCALE

    def lock(
        self,
        image: np.ndarray,
        tip: Sequence[float],
        throat: Sequence[float],
        t: float,
        exclude: Sequence[np.ndarray] = (),
    ) -> RacketPose:
        """Start tracking from the clicked tip and throat (full-resolution pixels)."""
        self.reset()
        self._tip = np.asarray(tip, dtype=float)
        self._throat = np.asarray(throat, dtype=float)
        self._prev = _small_grey(image)
        self._pts = self._detect(self._prev, exclude)
        self._last_good_t = t
        return self._pose(t)

    def update(self, image: np.ndarray, t: float, exclude: Sequence[np.ndarray] = ()) -> RacketPose:
        """Follow the racket into this frame. ``exclude``: hand landmark arrays (pixels)
        whose areas get no new features."""
        if self._tip is None or self._prev is None:
            return replace_t(NO_POSE, t)
        small = _small_grey(image)
        moved = self._track(small)
        self._prev = small
        if moved:
            self._last_good_t = t
        if len(self._pts) < MIN_POINTS:
            self._top_up(small, exclude)
        if t - self._last_good_t > self.cfg.hold_ms / 1000.0:
            pose = self._pose(t)
            return RacketPose(t, False, pose.origin, pose.u, pose.v, pose.head_len)
        return self._pose(t)

    # -- internals ----------------------------------------------------------------------

    def _track(self, small: np.ndarray) -> bool:
        if len(self._pts) < MIN_INLIERS:
            return False
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev, small, self._pts, None, **_LK)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(small, self._prev, nxt, None, **_LK)
        fb = np.linalg.norm((back - self._pts).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < FB_MAX_PX)
        if good.sum() < MIN_INLIERS:
            self._pts = nxt[good]
            return False
        src, dst = self._pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
        m, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX, maxIters=500
        )
        if m is None or inl is None or int(inl.sum()) < MIN_INLIERS:
            self._pts = nxt[good]
            return False
        scale = math.hypot(m[0, 0], m[1, 0])
        if abs(scale - 1.0) > MAX_SCALE_STEP:
            return False
        keep = inl.ravel().astype(bool)
        self._pts = dst[keep].reshape(-1, 1, 2).astype(np.float32)
        assert self._tip is not None and self._throat is not None
        self._tip = _apply(m, self._tip)
        self._throat = _apply(m, self._throat)
        return True

    def _top_up(self, small: np.ndarray, exclude: Sequence[np.ndarray]) -> None:
        fresh = self._detect(small, exclude, avoid=self._pts)
        if len(fresh):
            self._pts = np.concatenate([self._pts.reshape(-1, 1, 2), fresh]).astype(np.float32)

    def _detect(
        self, small: np.ndarray, exclude: Sequence[np.ndarray], avoid: np.ndarray | None = None
    ) -> np.ndarray:
        mask = self._ring_mask(small.shape)
        for lm in exclude:
            hull = cv2.convexHull((np.asarray(lm) * SCALE).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 0, lineType=cv2.LINE_8)
            cv2.polylines(mask, [hull], True, 0, 12)  # and a margin around it
        if avoid is not None:
            for x, y in avoid.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), 5, 0, -1)
        want = MAX_POINTS - (0 if avoid is None else len(avoid))
        if want <= 0:
            return np.zeros((0, 1, 2), np.float32)
        pts = cv2.goodFeaturesToTrack(
            small, maxCorners=want, qualityLevel=0.01, minDistance=5, mask=mask, blockSize=5
        )
        return np.zeros((0, 1, 2), np.float32) if pts is None else pts.astype(np.float32)

    def _ring_mask(self, shape: tuple[int, ...]) -> np.ndarray:
        assert self._tip is not None and self._throat is not None
        mask = np.zeros(shape[:2], np.uint8)
        d = self._tip - self._throat
        length = float(np.hypot(*d))
        if length < 1e-6:
            return mask
        u = d / length
        v = np.array([-u[1], u[0]])
        centre = (self._throat + 0.5 * d) * SCALE
        a = np.linspace(0, 2 * np.pi, 72)
        for frac, value in ((RING[1], 255), (RING[0], 0)):
            ring = (
                centre
                + np.outer(np.cos(a), u) * 0.5 * frac * length * SCALE
                + np.outer(np.sin(a), v) * self.head_half_width * frac * length * SCALE
            )
            cv2.fillPoly(mask, [ring.astype(np.int32)], value)
        return mask

    def _pose(self, t: float) -> RacketPose:
        assert self._tip is not None and self._throat is not None
        pose = pose_from_points(self._throat, self._tip, t)
        if pose is None:
            return replace_t(NO_POSE, t)
        return pose


def pose_from_points(throat: np.ndarray, tip: np.ndarray, t: float) -> RacketPose | None:
    """Pose from the throat and head-tip points (pixels), or None if they coincide."""
    d = np.asarray(tip, dtype=float) - np.asarray(throat, dtype=float)
    head_len = math.hypot(d[0], d[1])
    if head_len < 1e-6:
        return None
    u = Vec2(float(d[0] / head_len), float(d[1] / head_len))
    return RacketPose(
        t=float(t),
        visible=True,
        origin=Vec2(float(throat[0]), float(throat[1])),
        u=u,
        v=Vec2(-u.y, u.x),
        head_len=head_len,
    )


def replace_t(pose: RacketPose, t: float) -> RacketPose:
    return RacketPose(t, pose.visible, pose.origin, pose.u, pose.v, pose.head_len)


def _small_grey(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(grey, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def _apply(m: np.ndarray, p_full: np.ndarray) -> np.ndarray:
    p = p_full * SCALE
    return (m[:, :2] @ p + m[:, 2]) / SCALE
