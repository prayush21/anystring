"""The strum gate: pick-grip check (default) and optional mic onset window.

``make_gate(cfg, calibration)`` returns a ``Gate`` implementing ``StrumGate``:

* ``gate.pick_grip`` on: a crossing needs the strumming hand's grip to be ``"closed"``.
* ``gate.mic`` on: a crossing needs a recent contact onset (``OnsetWindow``).
* both on: both are needed; both off: every crossing plays (free air play).

**Grip score** (``grip_score``): the mean of two distances in palm sizes (wrist to
middle-finger MCP), so it ignores hand scale and rotation in the image:

* thumb gap: thumb tip (4) to the nearest point of the index finger (5-6-7-8), which is
  ~0.06 in a pick grip and ~0.7 with the thumb spread;
* finger reach: mean tip-to-MCP distance of the middle, ring and pinky fingers, ~0.25
  curled and ~0.9 extended.

Lower is more closed, as ``Calibration`` expects. A fist scores like a pick grip
(~0.16) and so arms the gate: the PID only separates a closed grip from an open hand,
and opening the hand is what disarms. The score uses ``HandObs.raw`` so that fingertip
smoothing never delays a disarm; hysteresis and the frame hold do the smoothing.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import replace
from itertools import pairwise

import numpy as np

from racketguitar.config import Calibration, Config, GateConfig
from racketguitar.types import GripState, HandObs, Onset

__all__ = [
    "Gate",
    "GripClassifier",
    "OnsetWindow",
    "make_gate",
    "grip_score",
    "grip_thresholds",
    "grip_calibrated",
    "hand_size",
    "calibrate_grip",
    "GRIP_SCORE_VERSION",
]

TIME_EPS = 1e-9  # seconds; window boundaries are inclusive despite float rounding
KEEP_S = 1.0  # onsets and notes older than this (beyond every window) are dropped

_WRIST, _THUMB_TIP, _MIDDLE_MCP = 0, 4, 9
_INDEX = (5, 6, 7, 8)
_REACH = ((9, 12), (13, 16), (17, 20))  # (MCP, tip) of middle, ring, pinky
_SIZE_BONES = ((0, 5), (0, 9), (0, 13), (0, 17), (5, 17))
# Bumped whenever grip_score changes: calibrations recorded with another version are
# ignored (and the app recalibrates), since their scores are on a different scale.
GRIP_SCORE_VERSION = 2


def _landmarks(hand: HandObs | np.ndarray) -> np.ndarray:
    return np.asarray(hand.raw if isinstance(hand, HandObs) else hand, dtype=float)


def hand_size(lm: np.ndarray) -> float:
    """The palm's largest bone (wrist to the index, middle, ring and pinky knuckles, and
    across the knuckles). Unlike wrist -> middle knuckle alone it stays near the true
    size when the palm turns edge-on to the camera, so the score does not spike."""
    return max(float(np.hypot(*(lm[a] - lm[b]))) for a, b in _SIZE_BONES)


def grip_score(hand: HandObs | np.ndarray) -> float:
    """Scalar grip score of one hand (lower = more closed); NaN if the palm is degenerate."""
    lm = _landmarks(hand)
    if not np.all(np.isfinite(lm)):
        return math.nan
    palm = hand_size(lm)
    if not palm > 1e-6:
        return math.nan
    gap = min(_point_segment(lm[_THUMB_TIP], lm[i], lm[j]) for i, j in pairwise(_INDEX))
    reach = sum(float(np.hypot(*(lm[tip] - lm[mcp]))) for mcp, tip in _REACH) / len(_REACH)
    return 0.5 * (gap + reach) / palm


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    s = min(max(float(np.dot(p - a, ab)) / max(float(np.dot(ab, ab)), 1e-12), 0.0), 1.0)
    return float(np.hypot(*(p - (a + s * ab))))


def grip_calibrated(cal: Calibration) -> bool:
    """Whether ``cal`` holds grip scores this version of ``grip_score`` can use."""
    return cal.grip_closed_score is not None and cal.grip_version == GRIP_SCORE_VERSION


def grip_thresholds(
    calibration: Calibration | None = None, cfg: GateConfig | None = None
) -> tuple[float, float]:
    """``(close_below, open_above)`` from the calibrated closed/open scores, else the
    ``GateConfig`` defaults (template pick grip ~0.16, open hand ~0.80): the
    ``grip_arm_below`` and ``grip_release_above`` fractions of the way from closed to open.

    A calibration whose closed score is not below its open score, or that was recorded
    with another ``GRIP_SCORE_VERSION``, is ignored.
    """
    cfg = cfg or GateConfig()
    c, o = cfg.grip_closed_default, cfg.grip_open_default
    if calibration is not None and grip_calibrated(calibration):
        cc, oo = calibration.grip_closed_score, calibration.grip_open_score
        if cc is not None and oo is not None and cc < oo:
            c, o = cc, oo
    span = o - c
    return c + cfg.grip_arm_below * span, c + cfg.grip_release_above * span


def calibrate_grip(
    closed: Iterable[HandObs | np.ndarray],
    open_: Iterable[HandObs | np.ndarray],
    base: Calibration | None = None,
) -> Calibration:
    """Turn recorded closed-grip and open-hand frames into ``Calibration`` grip scores.

    Each score is the median over its frames, so a few frames with a hidden thumb do not
    move it. Raises ``ValueError`` if either set is empty or closed is not below open.
    """
    scores = []
    for name, frames in (("closed", closed), ("open", open_)):
        s = np.array([grip_score(f) for f in frames], dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            raise ValueError(f"no usable {name} grip frames")
        scores.append(float(np.median(s)))
    c, o = scores
    if not c < o:
        raise ValueError(f"closed grip score {c:.3f} is not below open score {o:.3f}")
    return replace(
        base or Calibration(),
        grip_closed_score=c,
        grip_open_score=o,
        grip_version=GRIP_SCORE_VERSION,
    )


class GripClassifier:
    """Per-frame grip state with hysteresis and a frame hold. Not thread-safe.

    Each frame votes ``"closed"``, ``"open"`` (with hysteresis around the current state)
    or ``"unknown"`` (no hand). The state flips only after ``hold_frames`` consecutive
    votes for the same new state, so a single bad frame never arms the gate. Releasing
    a closed grip is the exception: it takes ``release_frames`` (default 1), so a hand
    opening for the return stroke disarms before that stroke reaches the strings.
    """

    def __init__(
        self,
        hold_frames: int = 2,
        calibration: Calibration | None = None,
        cfg: GateConfig | None = None,
        release_frames: int = 1,
    ) -> None:
        self.hold_frames = max(1, hold_frames)
        self.release_frames = max(1, release_frames)
        self.close_below, self.open_above = grip_thresholds(calibration, cfg)
        self.state: GripState = "unknown"
        self.score: float | None = None  # latest score, None when no hand
        self._pending: GripState | None = None
        self._count = 0

    def update(self, hand: HandObs | None) -> GripState:
        s = grip_score(hand) if hand is not None else math.nan
        self.score = s if math.isfinite(s) else None
        vote: GripState
        if self.score is None:
            vote = "unknown"
        elif self.state == "closed":
            vote = "open" if s > self.open_above else "closed"
        elif self.state == "open":
            vote = "closed" if s < self.close_below else "open"
        else:
            vote = "closed" if s < 0.5 * (self.close_below + self.open_above) else "open"
        if vote == self.state:
            self._pending, self._count = None, 0
        elif vote == self._pending:
            self._count += 1
        else:
            self._pending, self._count = vote, 1
        releasing = self.state == "closed" and vote == "open"
        if self._count >= (self.release_frames if releasing else self.hold_frames):
            self.state, self._pending, self._count = vote, None, 0
        return self.state


class OnsetWindow:
    """The mic half of the gate. Not thread-safe; tolerant of call order.

    * A crossing at ``t`` passes if a usable onset lies in ``[t - lookback, t + after]``.
      ``after`` absorbs onset-timing and crossing-interpolation error, so a contact
      measured a few ms after its camera crossing still counts.
    * Hold: the newest such onset becomes the sweep anchor once a crossing passes, and
      later crossings pass until ``anchor + hold`` even with no new onset, so one
      onset opens the gate for a whole sweep. A stray onset that admitted no crossing
      opens nothing beyond its own lookback.
    * Bleed: an onset is unusable if it falls in ``[tn, tn + latency + bleed_ignore]``
      for an emitted note ``tn``. ``tn`` is the camera detect time; the sound leaves
      the speaker about ``latency`` later, so the ignore window starts at ``tn`` and
      ends ``bleed_ignore`` after the expected sound.

    Onsets and notes are matched when ``allows`` runs, so an onset that is reported
    before (or after) the note that caused it is filtered either way. Onsets later than
    the queried ``t`` are kept for later queries. Boundaries are inclusive.
    """

    def __init__(
        self,
        lookback_ms: float = 80.0,
        hold_ms: float = 150.0,
        bleed_ignore_ms: float = 25.0,
        latency_ms: float = GateConfig.output_latency_ms,
        after_ms: float = 0.0,
    ) -> None:
        self.lookback = lookback_ms / 1000
        self.after = after_ms / 1000
        self.hold = hold_ms / 1000
        self.ignore_after = (latency_ms + bleed_ignore_ms) / 1000
        self._onsets: deque[float] = deque(maxlen=512)
        self._notes: deque[float] = deque(maxlen=512)
        self.anchor: float | None = None

    def add_onset(self, t: float) -> None:
        self._onsets.append(t)

    def add_note(self, t: float) -> None:
        self._notes.append(t)

    def is_bleed(self, t_onset: float) -> bool:
        return any(
            tn - TIME_EPS <= t_onset <= tn + self.ignore_after + TIME_EPS for tn in self._notes
        )

    def allows(self, t: float, *, commit: bool = True) -> bool:
        """Whether a crossing at ``t`` passes; ``commit`` records the sweep anchor."""
        self._prune(t)
        best = None
        for to in self._onsets:
            if t - self.lookback - TIME_EPS <= to <= t + self.after + TIME_EPS and not (
                self.is_bleed(to)
            ):
                if best is None or to > best:
                    best = to
        if best is not None:
            if commit and (self.anchor is None or best > self.anchor):
                self.anchor = best
            return True
        return (
            self.anchor is not None
            and self.anchor - TIME_EPS <= t <= self.anchor + self.hold + TIME_EPS
        )

    def _prune(self, t: float) -> None:
        cut = t - KEEP_S
        for q in (self._onsets, self._notes):
            while q and q[0] < cut:
                q.popleft()


class Gate:
    """``StrumGate`` combining the pick-grip check and the mic window per ``GateConfig``.

    ``allows(t)`` and ``note_emitted(t)`` take interpolated crossing times, which may be
    up to a frame earlier than the latest ``observe_hand`` t: the grip check uses the
    state as of the latest frame, and the mic window judges each ``t`` by its own time.
    """

    def __init__(
        self, cfg: GateConfig | None = None, calibration: Calibration | None = None
    ) -> None:
        self.cfg = cfg or GateConfig()
        cal = calibration or Calibration()
        latency = (
            cal.detect_to_audio_ms
            if cal.detect_to_audio_ms is not None
            else self.cfg.output_latency_ms
        )
        self.classifier = GripClassifier(
            self.cfg.grip_hold_frames, cal, self.cfg, self.cfg.grip_release_frames
        )
        self.window = OnsetWindow(
            self.cfg.onset_lookback_ms,
            self.cfg.onset_hold_ms,
            self.cfg.bleed_ignore_ms,
            latency,
            self.cfg.onset_after_ms,
        )

    @property
    def pick_grip(self) -> bool:
        return self.cfg.pick_grip

    @property
    def mic(self) -> bool:
        return self.cfg.mic

    def observe_hand(self, t: float, hand: HandObs | None) -> None:
        self.classifier.update(hand)

    def observe_onset(self, onset: Onset) -> None:
        if self.cfg.mic:
            self.window.add_onset(onset.t)

    def note_emitted(self, t: float) -> None:
        if self.cfg.mic:
            self.window.add_note(t)

    def allows(self, t: float) -> bool:
        if self.cfg.pick_grip and self.classifier.state != "closed":
            return False
        return not self.cfg.mic or self.window.allows(t)

    @property
    def grip(self) -> GripState:
        return self.classifier.state

    @property
    def grip_score(self) -> float | None:
        """Latest raw grip score, for the calibration screen; None when no hand."""
        return self.classifier.score


def make_gate(
    cfg: Config | GateConfig | None = None, calibration: Calibration | None = None
) -> Gate:
    """Build the strum gate from settings (a whole ``Config`` or its ``gate`` section)."""
    gate_cfg = cfg.gate if isinstance(cfg, Config) else cfg
    return Gate(gate_cfg, calibration)
