"""Palm mute: an open strumming hand resting on the head damps the strings.

The hand counts as resting when all four fingers are extended, its palm centre lies
inside the head, and the palm moves slower than ``MuteConfig.max_speed`` (racket frame,
head lengths/s). After ``MuteConfig.hold_ms`` of rest the mute fires once. It re-arms
when the rest breaks: the hand closes, leaves the head, moves fast, or is lost.

Openness is checked here from landmark distances (independent of the gate's pick-grip
classifier): a finger is extended when its tip is at least ``ratio``
times as far from the wrist as its MCP knuckle is (``MuteConfig.open_extension_ratio``).
"""

from __future__ import annotations

import numpy as np

from racketguitar.config import MuteConfig, StringsConfig
from racketguitar.geometry.racket import RacketFrame, in_head
from racketguitar.types import HandObs

__all__ = ["PALM_LANDMARKS", "is_open_hand", "palm_centre", "PalmMute"]

PALM_LANDMARKS = (0, 5, 9, 13, 17)  # wrist and the four MCP knuckles
_MCP = np.array([5, 9, 13, 17])
_TIP = _MCP + 3


def is_open_hand(landmarks: np.ndarray, ratio: float = MuteConfig.open_extension_ratio) -> bool:
    """All four fingers extended (the thumb is ignored). A straight finger is ~2.0,
    a curled one ~0.7-0.9 (MediaPipe-style landmarks, 2D)."""
    wrist = landmarks[0]
    tip = np.hypot(*(landmarks[_TIP] - wrist).T)
    mcp = np.hypot(*(landmarks[_MCP] - wrist).T)
    return bool(np.all(tip >= ratio * mcp))


def palm_centre(landmarks: np.ndarray) -> np.ndarray:
    return landmarks[list(PALM_LANDMARKS)].mean(axis=0)


class PalmMute:
    """Feed the strumming hand once per frame; ``update`` is True on the frame it fires."""

    def __init__(
        self, cfg: MuteConfig, head_half_width: float = StringsConfig.head_half_width
    ) -> None:
        self._max_speed = cfg.max_speed
        self._ratio = cfg.open_extension_ratio
        self._half_width = head_half_width
        self._hold = cfg.hold_ms / 1000.0
        self._prev: np.ndarray | None = None
        self._prev_t = 0.0
        self._rest_since: float | None = None
        self._fired = False

    @property
    def muted(self) -> bool:
        """True from the frame the mute fires until it re-arms."""
        return self._fired

    @property
    def resting(self) -> bool:
        return self._rest_since is not None

    def reset(self) -> None:
        self._prev = None
        self._rest_since = None
        self._fired = False

    def update(self, frame: RacketFrame | None, hand: HandObs | None) -> bool:
        if frame is None or hand is None:
            self.reset()
            return False
        palm = frame.to_uv(palm_centre(hand.landmarks))
        prev, prev_t = self._prev, self._prev_t
        self._prev, self._prev_t = palm, hand.t
        if not (is_open_hand(hand.landmarks, self._ratio) and in_head(palm, self._half_width)):
            self._rest_since = None
            self._fired = False
            return False
        dt = hand.t - prev_t
        if prev is None or dt <= 0.0:
            return False  # no speed yet
        if float(np.hypot(*(palm - prev))) / dt >= self._max_speed:
            self._rest_since = None
            self._fired = False
            return False
        if self._rest_since is None:
            self._rest_since = prev_t  # resting since the previous sample
        if not self._fired and hand.t - self._rest_since >= self._hold - 1e-9:
            self._fired = True
            return True
        return False
