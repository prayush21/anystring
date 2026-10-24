"""Rendered camera frames of the two racket markers, with exact ground truth.

The racket is modelled as a plane seen by a pinhole camera, so scenes can include
perspective tilt, not just in-plane rotation and scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from racketguitar.config import MarkerConfig
from racketguitar.types import RacketPose, Vec2

FOCAL_PX = 1000.0
HEAD_LEN_MM = 300.0  # throat marker centre to tip marker centre
QUIET_ZONE = 0.25  # white margin around each marker, as a fraction of its side


@dataclass(frozen=True)
class MarkerScene:
    image: np.ndarray  # (H, W, 3) uint8 BGR, like an OpenCV capture
    pose: RacketPose  # ground truth, t = 0
    corners: dict[int, np.ndarray]  # marker id -> (4, 2) pixel corners, ArUco order


def render_marker_scene(
    *,
    size: tuple[int, int] = (1280, 720),
    center_px: tuple[float, float] = (640.0, 360.0),
    angle_deg: float = -90.0,
    head_len_px: float = 300.0,
    marker_mm: float = 40.0,
    tilt_across_deg: float = 0.0,
    tilt_along_deg: float = 0.0,
    blur_sigma: float = 0.0,
    noise_std: float = 0.0,
    cfg: MarkerConfig | None = None,
    rng: np.random.Generator | None = None,
) -> MarkerScene:
    """Render the tip and throat markers.

    ``center_px`` is where the midpoint between the markers lands, ``angle_deg`` the
    image direction from throat to tip (-90 = up), and ``head_len_px`` the marker-centre
    distance before tilt. ``tilt_across_deg`` rotates the racket about its long axis
    (squashes v); ``tilt_along_deg`` about its short axis (one marker nearer the camera).

    Ground-truth marker centres are the projections of the true centres, i.e. the
    intersection of each projected square's diagonals, not the mean of its corners.
    """
    cfg = cfg or MarkerConfig()
    rng = rng or np.random.default_rng(0)
    w, h = size
    dist = FOCAL_PX * HEAD_LEN_MM / head_len_px

    ta, tl, rz = (math.radians(x) for x in (tilt_across_deg, tilt_along_deg, angle_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(ta), -math.sin(ta)], [0, math.sin(ta), math.cos(ta)]])
    ry = np.array([[math.cos(tl), 0, math.sin(tl)], [0, 1, 0], [-math.sin(tl), 0, math.cos(tl)]])
    rot_z = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    rot = rot_z @ rx @ ry

    def project(plane_mm: np.ndarray) -> np.ndarray:
        p = np.c_[plane_mm, np.zeros(len(plane_mm))] @ rot.T + [0.0, 0.0, dist]
        return np.c_[FOCAL_PX * p[:, 0] / p[:, 2], FOCAL_PX * p[:, 1] / p[:, 2]] + center_px

    img = np.full((h, w, 3), 110, np.uint8)
    # Faint head outline so the scene is not a pure flat field.
    head_ring = np.array(
        [(170 * math.cos(a), 125 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 64)]
    )
    cv2.polylines(img, [project(head_ring).astype(np.int32)], True, (70, 70, 70), 3, cv2.LINE_AA)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.dictionary))
    centres_mm = {cfg.throat_id: (-HEAD_LEN_MM / 2, 0.0), cfg.tip_id: (HEAD_LEN_MM / 2, 0.0)}
    side_px = 200
    pad = round(side_px * QUIET_ZONE)
    tile = np.full((side_px + 2 * pad,) * 2, 255, np.uint8)
    corners: dict[int, np.ndarray] = {}
    centres_px: dict[int, np.ndarray] = {}
    for marker_id, (cx, cy) in centres_mm.items():
        tile[pad:-pad, pad:-pad] = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
        half, outer = marker_mm / 2, marker_mm / 2 * (1 + 2 * QUIET_ZONE)
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left in the marker's
        # own image, with the marker's "up" pointing toward +u on the racket.
        sq = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], float)
        local = np.c_[-sq[:, 1], sq[:, 0]]  # marker image y-up = racket +u
        dst_outer = project(local * outer + (cx, cy))
        src = np.array(
            [(0, 0), (tile.shape[1], 0), tile.shape[::-1], (0, tile.shape[0])], np.float32
        )
        hom = cv2.getPerspectiveTransform(src, dst_outer.astype(np.float32))
        warped = cv2.warpPerspective(tile, hom, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(tile, 255), hom, (w, h), flags=cv2.INTER_LINEAR)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        img = (img * (1 - alpha) + warped[..., None] * alpha).astype(np.uint8)
        corners[marker_id] = project(local * half + (cx, cy))
        centres_px[marker_id] = project(np.array([[cx, cy]]))[0]

    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise_std > 0:
        img = np.clip(img + rng.normal(0, noise_std, img.shape), 0, 255).astype(np.uint8)

    origin, tip = centres_px[cfg.throat_id], centres_px[cfg.tip_id]
    d = tip - origin
    head_len = float(np.hypot(*d))
    u = Vec2(*(d / head_len))
    pose = RacketPose(
        t=0.0, visible=True, origin=Vec2(*origin), u=u, v=Vec2(-u.y, u.x), head_len=head_len
    )
    return MarkerScene(image=img, pose=pose, corners=corners)
