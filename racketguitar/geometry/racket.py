"""The racket's own 2D frame: pixels <-> (u, v) in head lengths, strings and handle.

Every play detector works in this frame, so racket motion and distance to the camera
cancel out. ``px = origin + (u * U + v * V) * head_len`` with ``v = (-u.y, u.x)``.
The overlay uses the same ``to_px`` projections, so what it draws is what is tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from racketguitar.config import ChordConfig, StringsConfig
from racketguitar.types import RacketPose

__all__ = [
    "RacketFrame",
    "StringLayout",
    "HandleAxis",
    "StringOrientation",
    "in_head",
]


@dataclass(frozen=True, eq=False)
class RacketFrame:
    """One frame's racket transform. ``basis`` rows are the unit u and v pixel vectors."""

    t: float
    origin: np.ndarray  # (2,) pixels
    basis: np.ndarray  # (2, 2): [u, v]
    head_len: float  # pixels

    @classmethod
    def from_pose(cls, pose: RacketPose) -> RacketFrame:
        """Build from a visible pose. ``v`` is recomputed from ``u`` per the convention."""
        u = np.asarray(pose.u, dtype=float)
        u = u / np.hypot(u[0], u[1])
        basis = np.array([u, (-u[1], u[0])])
        return cls(pose.t, np.asarray(pose.origin, dtype=float), basis, float(pose.head_len))

    @property
    def u(self) -> np.ndarray:
        return self.basis[0]

    @property
    def v(self) -> np.ndarray:
        return self.basis[1]

    def to_uv(self, px: np.ndarray) -> np.ndarray:
        """Pixels (..., 2) -> racket frame (..., 2) as (u, v) head lengths."""
        return ((np.asarray(px, dtype=float) - self.origin) / self.head_len) @ self.basis.T

    def to_px(self, uv: np.ndarray) -> np.ndarray:
        """Racket frame (..., 2) -> pixels (..., 2)."""
        return self.origin + (np.asarray(uv, dtype=float) @ self.basis) * self.head_len


def in_head(uv: np.ndarray, half_width: float = StringsConfig.head_half_width) -> bool:
    """Whether a racket-frame point lies inside the head ellipse (centre u = 0.5,
    semi-axes 0.5 along u and ``half_width`` across v)."""
    du = (uv[0] - 0.5) / 0.5
    dv = uv[1] / half_width
    return bool(du * du + dv * dv <= 1.0)


@dataclass(frozen=True, eq=False)
class StringLayout:
    """String segments parallel to u. Slots run from ``v_min`` to ``v_max``; which end is
    string 0 (low E) depends on the orientation (``low_e_neg_v``)."""

    slot_v: np.ndarray  # (n,) v of each slot, ascending
    u_min: float
    u_max: float

    @classmethod
    def from_config(cls, cfg: StringsConfig) -> StringLayout:
        if cfg.count != len(cfg.tuning):
            raise ValueError("strings.count must match the length of strings.tuning")
        return cls(np.linspace(cfg.v_min, cfg.v_max, cfg.count), cfg.u_min, cfg.u_max)

    @property
    def count(self) -> int:
        return len(self.slot_v)

    def string_of_slot(self, slot: int, low_e_neg_v: bool) -> int:
        return slot if low_e_neg_v else self.count - 1 - slot

    def string_v(self, low_e_neg_v: bool) -> np.ndarray:
        """v of each string, indexed by string (0 = low E)."""
        return self.slot_v if low_e_neg_v else self.slot_v[::-1]

    def segments_uv(self, low_e_neg_v: bool) -> np.ndarray:
        """(n, 2, 2) segment endpoints in the racket frame, indexed by string."""
        sv = self.string_v(low_e_neg_v)
        seg = np.empty((self.count, 2, 2))
        seg[:, 0, 0], seg[:, 1, 0] = self.u_min, self.u_max
        seg[:, 0, 1] = seg[:, 1, 1] = sv
        return seg

    def segments_px(self, frame: RacketFrame, low_e_neg_v: bool) -> np.ndarray:
        """(n, 2, 2) segment endpoints in pixels, indexed by string."""
        return frame.to_px(self.segments_uv(low_e_neg_v))


@dataclass(frozen=True)
class HandleAxis:
    """The handle as a segment along u at v = 0: t = 0 at ``u_near``, 1 at ``u_far``."""

    u_near: float
    u_far: float

    @classmethod
    def from_config(cls, cfg: ChordConfig) -> HandleAxis:
        if cfg.handle_u_near == cfg.handle_u_far:
            raise ValueError("chords.handle_u_near and handle_u_far must differ")
        return cls(cfg.handle_u_near, cfg.handle_u_far)

    def t_of(self, uv: np.ndarray) -> float:
        """Unclamped handle position of a racket-frame point."""
        return float((uv[0] - self.u_near) / (self.u_far - self.u_near))

    def distance(self, uv: np.ndarray) -> float:
        """Distance in head lengths from a racket-frame point to the handle segment."""
        t = min(1.0, max(0.0, self.t_of(uv)))
        du = uv[0] - (self.u_near + t * (self.u_far - self.u_near))
        return float(np.hypot(du, uv[1]))

    def u_at(self, t: float | np.ndarray) -> float | np.ndarray:
        return self.u_near + np.asarray(t, dtype=float) * (self.u_far - self.u_near)

    def sections_uv(self, n: int) -> np.ndarray:
        """(n, 2, 2) equal sections along the axis, head -> butt, in the racket frame."""
        edges = self.u_at(np.linspace(0.0, 1.0, n + 1))
        seg = np.zeros((n, 2, 2))
        seg[:, 0, 0], seg[:, 1, 0] = edges[:-1], edges[1:]
        return seg

    def sections_px(self, frame: RacketFrame, n: int) -> np.ndarray:
        return frame.to_px(self.sections_uv(n))


class StringOrientation:
    """Which side of v holds low E, resolved from ``StringsConfig.low_e_side``.

    "auto" puts low E on the side that is higher in the image (smaller pixel y). The
    neg-v side is higher exactly when ``v.y > 0``. To avoid flicker the side flips only
    when ``v.y`` passes ``margin`` (``StringsConfig.auto_side_margin``) the other way;
    the first frame takes the sign (ties -> neg_v).
    """

    def __init__(
        self,
        side: Literal["auto", "neg_v", "pos_v"],
        margin: float = StringsConfig.auto_side_margin,
    ) -> None:
        if side not in ("auto", "neg_v", "pos_v"):
            raise ValueError(f"bad low_e_side: {side!r}")
        self._side = side
        self._margin = margin
        self._low_e_neg_v: bool | None = None if side == "auto" else side == "neg_v"

    @property
    def low_e_neg_v(self) -> bool:
        return True if self._low_e_neg_v is None else self._low_e_neg_v

    def update(self, frame: RacketFrame) -> bool:
        if self._side == "auto":
            vy = float(frame.v[1])
            if self._low_e_neg_v is None:
                self._low_e_neg_v = vy >= 0.0
            elif self._low_e_neg_v and vy < -self._margin:
                self._low_e_neg_v = False
            elif not self._low_e_neg_v and vy > self._margin:
                self._low_e_neg_v = True
        return self.low_e_neg_v
