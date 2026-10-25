"""Markerless racket tracking: lock on with clicks, then follow with optical flow.

``lock`` takes the head's tip and the throat (where the head meets the shaft) in
pixels, and optionally the handle end. Corner features are picked on a ring around the head's
rim, where a racket has unique detail (frame graphics, string ends) and the strumming
hand rarely is; the string mesh inside is skipped because its repeating grid fools
optical flow. With the handle end given, features are also picked on a narrow strip over the
shaft and handle: a badminton head is nearly round, so its rim pins the rotation only
weakly, and the handle is a long lever whose every degree of error the chord sections
feel.

Every feature keeps an *anchor*: where it sits in the lock frame. Each frame the
features are followed with pyramidal Lucas-Kanade, checked forward and backward, and
one similarity transform (rotation, uniform scale, shift) from the anchors to where
the features are now is fitted with RANSAC, so features on a passing hand or the
background are outvoted. The pose is that transform applied to the clicked tip and
throat. Fitting against fixed anchors, instead of chaining one frame-to-frame transform
onto the last, keeps small per-frame errors from adding up into drift, and a frame whose
fit fails leaves no lasting offset. Features inside a hand are dropped and never picked.

It can still drift during fast moves and cannot tell the racket left the frame (the
static background keeps tracking); re-lock to reset. Works at half resolution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from anystring.config import TrackerConfig
from anystring.types import RacketPose, Vec2

__all__ = ["FlowTracker", "NO_POSE", "pose_from_points", "fit_rim_half_width"]

# Before a lock, and whenever there is no racket to report.
NO_POSE = RacketPose(
    t=0.0, visible=False, origin=Vec2(0.0, 0.0), u=Vec2(0.0, -1.0), v=Vec2(1.0, 0.0), head_len=0.0
)

SCALE = 0.5  # track on a half-resolution copy
MAX_POINTS = 300
MIN_POINTS = 60  # top up with fresh features below this
# ... but only while the last frame kept at least this share of its features: a frame
# that loses most of them has lost the racket (a fast swing, or out of view), and fresh
# features picked around a lost pose would lock it onto the background.
MIN_KEPT = 0.5
MIN_INLIERS = 12
FB_MAX_PX = 1.0  # forward-backward error allowed, tracking pixels
RANSAC_PX = 1.5
MAX_SCALE_STEP = 0.08  # reject a frame whose fitted scale changes more than this
RING = (0.75, 1.15)  # feature ring, as fractions of the head ellipse
HANDLE_SHARE = 0.3  # of new features, at most this share go on the handle strip
HANDLE_HALF_WIDTH = 0.05  # handle strip half width, head lengths
HAND_MARGIN_PX = 6  # tracking pixels kept clear around each hand
RIM_SEARCH = (0.2, 0.6)  # head half widths tried when fitting the rim at lock
_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
_IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_NO_POINTS = np.zeros((0, 1, 2), np.float32)


class FlowTracker:
    """Not thread-safe; vision thread only."""

    def __init__(self, cfg: TrackerConfig | None = None, *, head_half_width: float = 0.4) -> None:
        self.cfg = cfg or TrackerConfig()
        self.head_half_width = head_half_width
        self.reset()

    def reset(self) -> None:
        self._tip: np.ndarray | None = None  # lock-frame tracking pixels
        self._throat: np.ndarray | None = None
        self._handle_end: np.ndarray | None = None
        self._half_width = self.head_half_width  # fitted to the rim at lock
        self._m = _IDENTITY.copy()  # lock frame -> current frame, tracking pixels
        self._prev: np.ndarray | None = None  # previous small grey frame
        self._pts = _NO_POINTS.copy()  # tracking pixels, current frame
        self._anchors = np.zeros((0, 2), np.float32)  # the same features, lock frame
        self._last_good_t = -math.inf
        self._healthy = True  # the last fit kept at least MIN_KEPT of its features

    @property
    def locked(self) -> bool:
        return self._tip is not None

    @property
    def points(self) -> np.ndarray:
        """Tracked feature points in full-resolution pixels, (n, 2), for the overlay."""
        return self._pts.reshape(-1, 2) / SCALE

    @property
    def rim_half_width(self) -> float:
        """Head half width (head lengths) the feature ring follows, fitted at lock."""
        return self._half_width

    def lock(
        self,
        image: np.ndarray,
        tip: Sequence[float],
        throat: Sequence[float],
        t: float,
        exclude: Sequence[np.ndarray] = (),
        handle_end: Sequence[float] | None = None,
    ) -> RacketPose:
        """Start tracking from the clicked tip, throat and (optionally) handle end, in
        full-resolution pixels."""
        self.reset()
        self._tip = np.asarray(tip, dtype=float) * SCALE
        self._throat = np.asarray(throat, dtype=float) * SCALE
        if handle_end is not None:
            self._handle_end = np.asarray(handle_end, dtype=float) * SCALE
        self._prev = _small_grey(image)
        fitted = fit_rim_half_width(self._prev, self._tip, self._throat)
        if fitted is not None:
            self._half_width = fitted
        self._top_up(self._prev, self._hand_mask(self._prev.shape, exclude))
        self._last_good_t = t
        return self._pose(t)

    def update(self, image: np.ndarray, t: float, exclude: Sequence[np.ndarray] = ()) -> RacketPose:
        """Follow the racket into this frame. ``exclude``: hand landmark arrays (pixels)
        whose areas are not tracked."""
        if self._tip is None or self._prev is None:
            return replace_t(NO_POSE, t)
        small = _small_grey(image)
        hands = self._hand_mask(small.shape, exclude)
        if self._track(small, hands):
            self._last_good_t = t
        self._prev = small
        if len(self._pts) < MIN_POINTS and self._healthy:
            self._top_up(small, hands)
        pose = self._pose(t)
        if t - self._last_good_t > self.cfg.hold_ms / 1000.0:
            return RacketPose(t, False, pose.origin, pose.u, pose.v, pose.head_len)
        return pose

    # -- internals ----------------------------------------------------------------------

    def _track(self, small: np.ndarray, hands: np.ndarray) -> bool:
        before = len(self._pts)
        self._healthy = False
        if before < MIN_INLIERS:
            return False
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self._prev, small, self._pts, None, **_LK)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(small, self._prev, nxt, None, **_LK)
        fb = np.linalg.norm((back - self._pts).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < FB_MAX_PX)
        good &= ~_in_mask(hands, nxt.reshape(-1, 2))
        # Followed features keep their anchors even when this frame's fit fails, so the
        # pose simply catches up on the next good frame.
        self._pts, self._anchors = nxt[good], self._anchors[good]
        if len(self._pts) < MIN_INLIERS:
            return False
        m, inl = cv2.estimateAffinePartial2D(
            self._anchors,
            self._pts.reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_PX,
            maxIters=500,
        )
        if m is None or inl is None or int(inl.sum()) < MIN_INLIERS:
            return False
        if abs(_scale(m) / _scale(self._m) - 1.0) > MAX_SCALE_STEP:
            return False
        keep = inl.ravel().astype(bool)
        self._pts, self._anchors = self._pts[keep], self._anchors[keep]
        self._m = m
        self._healthy = len(self._pts) >= MIN_KEPT * before
        return True

    def _top_up(self, small: np.ndarray, hands: np.ndarray) -> None:
        room = MAX_POINTS - len(self._pts)
        if room <= 0:
            return
        ring, strip = self._region_masks(small.shape)
        for mask in (ring, strip):
            mask[hands > 0] = 0
            for x, y in self._pts.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), 5, 0, -1)
        on_handle = _detect(small, strip, int(room * HANDLE_SHARE))
        fresh = np.concatenate([on_handle, _detect(small, ring, room - len(on_handle))])
        if not len(fresh):
            return
        anchors = _apply(cv2.invertAffineTransform(self._m), fresh.reshape(-1, 2))
        self._pts = np.concatenate([self._pts.reshape(-1, 1, 2), fresh]).astype(np.float32)
        self._anchors = np.concatenate([self._anchors, anchors]).astype(np.float32)

    def _region_masks(self, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """The rim ring and the handle strip in the current frame (strip empty without a
        handle end). Both are drawn in the lock frame and moved by the current transform."""
        assert self._tip is not None and self._throat is not None
        ring = np.zeros(shape[:2], np.uint8)
        strip = np.zeros(shape[:2], np.uint8)
        d = self._tip - self._throat
        length = float(np.hypot(*d))
        if length < 1e-6:
            return ring, strip
        u = d / length
        v = np.array([-u[1], u[0]])
        centre = self._throat + 0.5 * d
        a = np.linspace(0, 2 * np.pi, 72)
        for frac, value in ((RING[1], 255), (RING[0], 0)):
            outline = (
                centre
                + np.outer(np.cos(a), u) * 0.5 * frac * length
                + np.outer(np.sin(a), v) * self._half_width * frac * length
            )
            cv2.fillPoly(ring, [_apply(self._m, outline).astype(np.int32)], value)
        if self._handle_end is not None:
            w = v * HANDLE_HALF_WIDTH * length
            quad = np.array(
                [self._throat + w, self._handle_end + w, self._handle_end - w, self._throat - w]
            )
            cv2.fillPoly(strip, [_apply(self._m, quad).astype(np.int32)], 255)
            strip[ring > 0] = 0
        return ring, strip

    def _hand_mask(self, shape: tuple[int, ...], exclude: Sequence[np.ndarray]) -> np.ndarray:
        mask = np.zeros(shape[:2], np.uint8)
        for lm in exclude:
            hull = cv2.convexHull((np.asarray(lm) * SCALE).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_8)
            cv2.polylines(mask, [hull], True, 255, 2 * HAND_MARGIN_PX)
        return mask

    def _pose(self, t: float) -> RacketPose:
        assert self._tip is not None and self._throat is not None
        tip, throat = _apply(self._m, np.stack([self._tip, self._throat])) / SCALE
        pose = pose_from_points(throat, tip, t)
        if pose is None:
            return replace_t(NO_POSE, t)
        return pose


def fit_rim_half_width(
    grey: np.ndarray, tip: np.ndarray, throat: np.ndarray, search: tuple[float, float] = RIM_SEARCH
) -> float | None:
    """The head half width (head lengths) whose ellipse through ``tip`` and ``throat``
    best follows the rim: the strongest median edge across the ellipse's sides. None if
    the points coincide. Rackets differ (a badminton head is rounder than a tennis head),
    and a ring wider than the real head would pick background features."""
    d = np.asarray(tip, dtype=float) - np.asarray(throat, dtype=float)
    length = float(np.hypot(*d))
    if length < 1e-6:
        return None
    u = d / length
    v = np.array([-u[1], u[0]])
    centre = np.asarray(throat, dtype=float) + 0.5 * d
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    # The sides only: near the tip and throat every candidate ellipse runs together.
    side = np.linspace(np.pi / 6, 5 * np.pi / 6, 40)
    a = np.concatenate([side, side + np.pi])
    best, best_edge = None, -1.0
    for h in np.arange(search[0], search[1] + 1e-9, 0.01):
        p = centre + np.outer(np.cos(a), u) * 0.5 * length + np.outer(np.sin(a), v) * h * length
        n = np.outer(np.cos(a) / 0.5, u) + np.outer(np.sin(a) / h, v)  # outward normals
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        x = np.clip(np.round(p[:, 0]).astype(int), 0, grey.shape[1] - 1)
        y = np.clip(np.round(p[:, 1]).astype(int), 0, grey.shape[0] - 1)
        edge = float(np.median(np.abs(gx[y, x] * n[:, 0] + gy[y, x] * n[:, 1])))
        if edge > best_edge:
            best, best_edge = float(h), edge
    return best


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


def _detect(small: np.ndarray, mask: np.ndarray, want: int) -> np.ndarray:
    if want <= 0 or not mask.any():
        return _NO_POINTS.copy()
    pts = cv2.goodFeaturesToTrack(
        small, maxCorners=want, qualityLevel=0.01, minDistance=5, mask=mask, blockSize=5
    )
    return _NO_POINTS.copy() if pts is None else pts.astype(np.float32)


def _apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:, :2].T + m[:, 2]


def _scale(m: np.ndarray) -> float:
    return math.sqrt(abs(float(np.linalg.det(m[:, :2]))))


def _in_mask(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return mask[y, x] > 0
