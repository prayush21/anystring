"""Rendered camera frames of a plain racket (no markers), with exact ground truth.

The racket is a textured sprite (rim detail, string mesh, shaft, taped grip) placed by a
similarity transform on a static background, for testing the optical-flow racket tracker.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import cv2
import numpy as np

_SPRITE = 800  # sprite canvas, racket plane pixels
_TIP_S = np.array([400.0, 80.0])  # head top in the sprite
_THROAT_S = np.array([400.0, 440.0])  # where the head meets the shaft
_BUTT_S = np.array([400.0, 790.0])  # end of the grip
_HEAD_HALF_WIDTH = 150.0 / 360.0  # head half width over head length


@functools.cache
def _racket_sprite(seed: int, plain_rim: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A plain racket seen face-on, tip up: textured rim (or one plain colour), string
    mesh, shaft and a taped grip."""
    rng = np.random.default_rng(seed)
    img = np.zeros((_SPRITE, _SPRITE, 3), np.uint8)
    alpha = np.zeros((_SPRITE, _SPRITE), np.uint8)
    centre = tuple(((_TIP_S + _THROAT_S) / 2).astype(int))
    axes = (150, 180)
    # String mesh inside the head (a repeating grid: the hard case for optical flow).
    mesh = np.zeros_like(alpha)
    cv2.ellipse(mesh, centre, axes, 0, 0, 360, 255, -1)
    grid = np.zeros_like(img)
    grid[::10, :] = 235
    grid[:, ::10] = 235
    img[mesh > 0] = grid[mesh > 0]
    alpha[mesh > 0] = np.where(grid[mesh > 0, 0] > 0, 255, 0)
    # Rim with unique detail: random coloured blobs on a dark frame.
    rim = np.zeros_like(alpha)
    cv2.ellipse(rim, centre, (axes[0] + 9, axes[1] + 9), 0, 0, 360, 255, 18)
    tex = rng.integers(0, 256, (_SPRITE // 4, _SPRITE // 4, 3), dtype=np.uint8)
    tex = cv2.resize(tex, (_SPRITE, _SPRITE), interpolation=cv2.INTER_NEAREST)
    img[rim > 0] = (150, 60, 40) if plain_rim else tex[rim > 0]
    alpha[rim > 0] = 255
    # Shaft, then a wider grip with diagonal tape bands.
    cv2.line(img, (400, 455), (400, 790), (60, 40, 160), 9)
    cv2.line(alpha, (400, 455), (400, 790), 255, 9)
    grip = np.zeros_like(alpha)
    cv2.rectangle(grip, (391, 640), (409, 790), 255, -1)
    bands = (np.add.outer(np.arange(_SPRITE), np.arange(_SPRITE)) // 9) % 2  # row + col
    tape = np.where(bands[..., None] == 1, (30, 30, 30), (200, 200, 200)).astype(np.uint8)
    img[grip > 0] = tape[grip > 0]
    alpha[grip > 0] = 255
    return img, alpha


@functools.cache
def _background(size: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = size
    small = rng.integers(60, 200, (h // 40 + 1, w // 40 + 1, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


@dataclass(frozen=True)
class RacketScene:
    image: np.ndarray
    tip: np.ndarray  # pixels
    throat: np.ndarray
    head_half_width: float
    butt: np.ndarray  # pixels


def render_racket(
    *,
    size: tuple[int, int] = (1280, 720),
    center_px: tuple[float, float] = (640.0, 360.0),
    angle_deg: float = -90.0,
    head_len_px: float = 300.0,
    occluder: tuple[float, float, float] | None = None,
    show_racket: bool = True,
    plain_rim: bool = False,
    seed: int = 0,
) -> RacketScene:
    """A markerless racket (no markers) on a static textured background.

    ``center_px`` is where the head's centre lands, ``angle_deg`` the throat -> tip
    direction in the image. ``occluder`` = (x, y, radius) draws a hand-coloured disc on
    top, like a strumming hand passing over the head. ``plain_rim`` paints the rim one
    colour, so (as on a real round badminton head) it says little about rotation.
    """
    sprite, alpha = _racket_sprite(seed, plain_rim)
    img = _background(size, seed + 1).copy()
    s = head_len_px / float(np.hypot(*(_TIP_S - _THROAT_S)))
    rot = math.radians(angle_deg + 90.0)  # the sprite's tip points up (-90 deg)
    c, sn = math.cos(rot) * s, math.sin(rot) * s
    mid = (_TIP_S + _THROAT_S) / 2
    m = np.array([[c, -sn, 0.0], [sn, c, 0.0]])
    m[:, 2] = np.asarray(center_px) - m[:, :2] @ mid
    if show_racket:
        w, h = size
        warped = cv2.warpAffine(sprite, m, (w, h), flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(alpha, m, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
        img = (img * (1 - a[..., None]) + warped * a[..., None]).astype(np.uint8)
    if occluder is not None:
        x, y, r = occluder
        cv2.circle(img, (int(x), int(y)), int(r), (120, 150, 200), -1, cv2.LINE_AA)
    tip, throat, butt = (m[:, :2] @ p + m[:, 2] for p in (_TIP_S, _THROAT_S, _BUTT_S))
    return RacketScene(img, tip, throat, _HEAD_HALF_WIDTH, butt)
