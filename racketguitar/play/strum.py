"""Per-string strum detection from strumming fingertips, in the racket frame.

A hit is a sign change of a fingertip's signed v-distance to a string between two
consecutive samples, with the interpolated crossing point inside the string's u range
and perpendicular speed at least ``StrumConfig.min_speed``. Each sample is taken in its
own frame's racket coordinates, so racket motion cancels. A fast sweep that crosses
several strings between two frames yields one hit per string, ordered and timestamped
by linear interpolation along the path.

Two fingertips of the same hand (index and middle by default) pass each string during
one sweep; the second pass is a duplicate, not a new note. A crossing is dropped when
another tracked fingertip crossed the same string in the same direction, inside the
segment, after this fingertip's own previous pass over that string, and within
``StrumConfig.finger_dedupe_ms``. The per-string refractory window (``StrumConfig.refractory_ms``)
applies on top of that.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from racketguitar.config import StringsConfig, StrumConfig
from racketguitar.geometry.racket import RacketFrame, StringLayout
from racketguitar.types import HandObs, StringHit

__all__ = ["StrumDetector"]

# Tuning lives in StrumConfig:
# * finger_dedupe_ms: window in which a second fingertip's same-direction pass over a
#   string is a duplicate; covers a 0.12 head-length fingertip offset at min speed.
# * max_tip_speed: a fingertip moving faster than this (head lengths/s) between samples
#   is a tracking jump (lost hand, swapped identity), handled as re-acquired.
# * speed_window_ms: perpendicular speed is the net v displacement over the fingertip's
#   current monotonic run, looking back at most this far (always at least one frame).
#   Averaging a few frames keeps landmark jitter from pushing a real crossing under
#   min_speed, and stopping at a reversal keeps a sharp turn from reading as slow.
_HISTORY = 8


class StrumDetector:
    """Feed one strumming-hand sample per frame; get the string hits it produced.

    Not thread-safe; call from the vision thread only.
    """

    def __init__(
        self, strings: StringsConfig, strum: StrumConfig, skip_first_frames_back: int = 1
    ) -> None:
        self._layout = StringLayout.from_config(strings)
        self._tips = np.asarray(strum.fingertips, dtype=int)
        self._min_speed = strum.min_speed
        self._refractory = strum.refractory_ms / 1000.0
        self._dedupe = strum.finger_dedupe_ms / 1000.0
        self._window = strum.speed_window_ms / 1000.0
        self._max_tip_speed = strum.max_tip_speed
        self._skip = max(0, skip_first_frames_back)
        n_tips, n_str = len(self._tips), self._layout.count
        self._last_hit_t = np.full(n_str, -np.inf)  # by string
        self._cross_dir = np.zeros((n_tips, n_str), dtype=np.int8)  # by tip, slot
        self._cross_t = np.full((n_tips, n_str), -np.inf)
        self._others = ~np.eye(n_tips, dtype=bool)
        self._prev_uv: np.ndarray | None = None
        self._prev_t = 0.0
        self._hist: deque[tuple[float, np.ndarray]] = deque(maxlen=_HISTORY)  # (t, v per tip)
        self._frames_back = 0
        self._tips_uv: np.ndarray | None = None

    @property
    def layout(self) -> StringLayout:
        return self._layout

    @property
    def tips_uv(self) -> np.ndarray | None:
        """Latest fingertip positions in the racket frame, (k, 2), or None."""
        return self._tips_uv

    def reset(self) -> None:
        """Forget the hand, as if it had been lost."""
        self._prev_uv = None
        self._hist.clear()
        self._frames_back = 0
        self._tips_uv = None
        self._cross_dir[:] = 0
        self._cross_t[:] = -np.inf

    def update(
        self, frame: RacketFrame | None, hand: HandObs | None, low_e_neg_v: bool
    ) -> list[StringHit]:
        """Process one frame. ``frame`` is None while the racket is not visible."""
        if hand is None:
            self.reset()
            return []
        if frame is None:
            # Hand still seen, racket not: break the path but do not count a re-acquire.
            self._prev_uv = None
            self._hist.clear()
            self._tips_uv = None
            return []

        cur = frame.to_uv(hand.landmarks[self._tips])
        self._tips_uv = cur
        self._frames_back += 1
        if self._frames_back <= self._skip:
            return []  # first frame(s) back after a loss are not trusted
        prev, t0, t1 = self._prev_uv, self._prev_t, hand.t
        if prev is None:
            self._accept(cur, t1)
            return []
        dt = t1 - t0
        if dt <= 0.0:
            return []
        if float(np.max(np.hypot(*(cur - prev).T))) / dt > self._max_tip_speed:
            self.reset()
            self._tips_uv = cur
            self._frames_back = 1
            if self._skip < 1:
                self._accept(cur, t1)
            return []
        self._accept(cur, t1)

        slots = self._layout.slot_v
        d0 = prev[:, 1:2] - slots  # (k, n) signed v-distance per tip and slot
        d1 = cur[:, 1:2] - slots
        crossed = (d0 >= 0.0) != (d1 >= 0.0)
        if not crossed.any():
            return []

        speed = [self._run_speed(k) if crossed[k].any() else 0.0 for k in range(len(cur))]
        cands = []
        for k, s in zip(*np.nonzero(crossed), strict=True):
            f = d0[k, s] / (d0[k, s] - d1[k, s])
            u = prev[k, 0] + f * (cur[k, 0] - prev[k, 0])
            direction = 1 if d1[k, s] > d0[k, s] else -1
            cands.append((t0 + f * dt, int(k), int(s), direction, u))
        cands.sort()

        hits: list[StringHit] = []
        for t, k, s, direction, u in cands:
            own_last = self._cross_t[k, s]
            self._cross_t[k, s] = t
            if not self._layout.u_min <= u <= self._layout.u_max:
                self._cross_dir[k, s] = 0  # passed the string's line outside the segment
                continue
            self._cross_dir[k, s] = direction
            # Same sweep: another tip crossed this way after this tip's own last pass.
            dup = np.any(
                (self._others[k])
                & (self._cross_dir[:, s] == direction)
                & (self._cross_t[:, s] > own_last)
                & (t - self._cross_t[:, s] <= self._dedupe)
            )
            if dup or speed[k] < self._min_speed:
                continue
            string = self._layout.string_of_slot(s, low_e_neg_v)
            if t - self._last_hit_t[string] < self._refractory:
                continue
            self._last_hit_t[string] = t
            down = (direction > 0) == low_e_neg_v
            hits.append(StringHit(float(t), string, "down" if down else "up", float(speed[k])))
        return hits

    def _accept(self, uv: np.ndarray, t: float) -> None:
        self._prev_uv, self._prev_t = uv, t
        self._hist.append((t, uv[:, 1].copy()))

    def _run_speed(self, k: int) -> float:
        """|dv/dt| of tip ``k`` over its current monotonic run within the window."""
        hist = self._hist
        t1, v1 = hist[-1][0], hist[-1][1][k]
        t_old, v_old = hist[-2][0], hist[-2][1][k]
        step = v1 - v_old
        for i in range(len(hist) - 3, -1, -1):
            t_i, v_i = hist[i][0], hist[i][1][k]
            if t1 - t_i > self._window or (v_old - v_i) * step <= 0.0:
                break
            t_old, v_old = t_i, v_i
        return abs(v1 - v_old) / (t1 - t_old)
