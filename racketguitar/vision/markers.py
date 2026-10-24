"""ArUco racket markers -> ``RacketPose``.

Marker ``tip_id`` sits at the head tip and ``throat_id`` at the throat. Each marker's
centre is the intersection of its corner diagonals (the projection of the true square
centre, which the corner mean is not under perspective). ``origin`` is the throat
centre, ``u`` the unit vector toward the tip centre and ``head_len`` their distance.

When either marker drops out the last pose is held for ``MarkerConfig.hold_ms`` (still
``visible=True``, same geometry, new ``t``); after that the tracker emits
``visible=False`` with the last known geometry, so the overlay can still draw it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np

from racketguitar.config import FilterConfig, MarkerConfig
from racketguitar.types import RacketPose, Vec2

from .filters import OneEuroFilter, marker_filter

__all__ = [
    "MarkerTracker",
    "make_detector",
    "detect_markers",
    "marker_centre",
    "pose_from_centres",
    "pose_from_corners",
    "NO_POSE",
]

# Detector tuning, chosen on the rendered scenes in tests/vision/test_markers.py.
# Contour refinement was the most accurate under blur; 8 px per cell with a 0.3 cell
# margin roughly halves the misses on small markers versus the OpenCV defaults.
CORNER_REFINEMENT = cv2.aruco.CORNER_REFINE_CONTOUR
PERSPECTIVE_PX_PER_CELL = 8
PERSPECTIVE_IGNORED_MARGIN = 0.3

# Emitted before any marker pair has ever been seen.
NO_POSE = RacketPose(
    t=0.0, visible=False, origin=Vec2(0.0, 0.0), u=Vec2(0.0, -1.0), v=Vec2(1.0, 0.0), head_len=0.0
)


def make_detector(cfg: MarkerConfig) -> cv2.aruco.ArucoDetector:
    """An ``ArucoDetector`` for the configured dictionary with this module's tuning."""
    try:
        dict_id = getattr(cv2.aruco, cfg.dictionary)
    except AttributeError:
        raise ValueError(f"unknown ArUco dictionary: {cfg.dictionary!r}") from None
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = CORNER_REFINEMENT
    params.perspectiveRemovePixelPerCell = PERSPECTIVE_PX_PER_CELL
    params.perspectiveRemoveIgnoredMarginPerCell = PERSPECTIVE_IGNORED_MARGIN
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)


def detect_markers(
    detector: cv2.aruco.ArucoDetector, image: np.ndarray, ids: tuple[int, ...] | None = None
) -> dict[int, np.ndarray]:
    """Marker id -> (4, 2) float corners in ArUco order (TL, TR, BR, BL of the marker).

    ``image`` is BGR or grayscale. If an id is found twice the larger quad wins. With
    ``ids`` given, other markers are dropped.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, found, _ = detector.detectMarkers(gray)
    out: dict[int, np.ndarray] = {}
    if found is None:
        return out
    for c, marker_id in zip(corners, found.ravel(), strict=True):
        mid = int(marker_id)
        if ids is not None and mid not in ids:
            continue
        quad = np.asarray(c, dtype=float).reshape(4, 2)
        if mid not in out or _area(quad) > _area(out[mid]):
            out[mid] = quad
    return out


def _area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def marker_centre(corners: np.ndarray) -> np.ndarray:
    """Intersection of the diagonals corner0-corner2 and corner1-corner3."""
    p0, p1, p2, p3 = np.asarray(corners, dtype=float).reshape(4, 2)
    a, b, r = p2 - p0, p3 - p1, p1 - p0
    denom = a[0] * b[1] - a[1] * b[0]
    if abs(denom) < 1e-9:  # degenerate quad; fall back to the mean
        return (p0 + p1 + p2 + p3) / 4.0
    s = (r[0] * b[1] - r[1] * b[0]) / denom
    return p0 + s * a


def pose_from_centres(throat: np.ndarray, tip: np.ndarray, t: float) -> RacketPose | None:
    """Pose from the two marker centres, or None if they coincide."""
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


def pose_from_corners(
    corners: dict[int, np.ndarray], t: float, cfg: MarkerConfig
) -> RacketPose | None:
    """Pose from detected corners, or None unless both markers are present."""
    if cfg.throat_id not in corners or cfg.tip_id not in corners:
        return None
    return pose_from_centres(
        marker_centre(corners[cfg.throat_id]), marker_centre(corners[cfg.tip_id]), t
    )


class MarkerTracker:
    """Per-frame marker detection, optional One Euro corner smoothing and pose hold.

    Not thread-safe; call from the vision thread with non-decreasing ``t``.
    """

    def __init__(
        self,
        cfg: MarkerConfig | None = None,
        filters: FilterConfig | None = None,
        *,
        smooth: bool | None = None,
    ) -> None:
        self.cfg = cfg or MarkerConfig()
        self.filter_cfg = filters or FilterConfig()
        self.smooth = self.cfg.smooth if smooth is None else smooth
        self.detector = make_detector(self.cfg)
        self._ids = (self.cfg.tip_id, self.cfg.throat_id)
        self._filters: dict[int, OneEuroFilter] = {}
        self._seen_t: dict[int, float] = {}  # marker id -> last time it was detected
        self._last: RacketPose | None = None  # last pose computed from both markers
        self._corners: dict[int, np.ndarray] = {}

    @property
    def corners(self) -> dict[int, np.ndarray]:
        """Corners found in the latest frame (smoothed if enabled), for the overlay."""
        return dict(self._corners)

    def reset(self) -> None:
        self._filters.clear()
        self._seen_t.clear()
        self._last = None
        self._corners = {}

    def update(self, image: np.ndarray, t: float) -> RacketPose:
        """Detect markers in a full-resolution frame captured at ``t``."""
        return self.update_corners(detect_markers(self.detector, image, self._ids), t)

    def update_corners(self, corners: dict[int, np.ndarray], t: float) -> RacketPose:
        """Advance the tracker with already detected corners (e.g. from a replay)."""
        hold_s = self.cfg.hold_ms / 1000.0
        found: dict[int, np.ndarray] = {}
        for mid in self._ids:
            if mid not in corners:
                continue
            quad = np.asarray(corners[mid], dtype=float).reshape(4, 2)
            if self.smooth:
                f = self._filters.get(mid)
                if f is None:
                    f = self._filters[mid] = marker_filter(self.filter_cfg)
                elif t - self._seen_t.get(mid, -math.inf) > hold_s:
                    f.reset()  # re-acquired after a long gap: do not glide from stale corners
                quad = f(quad, t)
            found[mid] = quad
            self._seen_t[mid] = t
        self._corners = found

        pose = pose_from_corners(found, t, self.cfg)
        if pose is not None:
            self._last = pose
            return pose
        if self._last is None:
            return replace(NO_POSE, t=float(t))
        held = t - self._last.t <= hold_s
        return replace(self._last, t=float(t), visible=held)
