"""Calibration flow: marker check, closed pick grip, open hand, and (if on) the mic.

A state machine fed once per frame from the vision thread, driven only by frame times,
so it runs the same live and in tests. The overlay draws ``instruction`` and
``progress``; when ``done``, ``result`` is the new ``Calibration`` (not yet saved).

Steps: markers -> closed -> open -> [quiet -> brush] -> done. A grip step waits for the
strumming hand to be seen continuously for ``SETTLE_S``, then records its frames for
``GRIP_CAPTURE_S``; losing the hand restarts that step. If the closed grip does not
score below the open hand, the flow returns to the closed step with an error.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from racketguitar.capture.mic import calibrate_mic, mic_margin
from racketguitar.config import Calibration, Config
from racketguitar.play.gate import calibrate_grip
from racketguitar.replay.record import TeeDetector
from racketguitar.types import HandObs, RacketPose

__all__ = ["Step", "CalibrationFlow", "INSTRUCTIONS"]

Step = Literal["markers", "closed", "open", "quiet", "brush", "done"]

MARKERS_HOLD_S = 1.0  # both markers seen continuously this long
SETTLE_S = 1.0  # strumming hand held still before a grip capture starts
GRIP_CAPTURE_S = 1.5
QUIET_S = 2.0
BRUSH_S = 4.0
MARKERLESS_LOCK = "Click the racket: top of the head, throat, grip start, butt"
# The spread hand must score at least this many times the pick grip, or the two are too
# alike to tell apart and the calibration is redone (a bad one arms all the time).
MIN_GRIP_RATIO = 1.5
LOW_MIC_MARGIN = 2.0  # warn when brushes are this close to the detection threshold

INSTRUCTIONS: dict[str, str] = {
    "markers": "Show both printed markers (markers.pdf) taped on the racket",
    "closed": "Hold your pick grip over the strings, exactly as you play",
    "open": "Now spread your strumming hand wide open, fingers straight and apart",
    "quiet": "Keep still and quiet",
    "brush": "Brush the strings about five times",
    "done": "Calibration done",
}


class CalibrationFlow:
    def __init__(
        self,
        cfg: Config,
        base: Calibration | None = None,
        mic: TeeDetector | None = None,
        *,
        markerless: bool = False,
    ) -> None:
        self.cfg = cfg
        self.markerless = markerless
        self.base = base or Calibration()
        self.mic = mic
        self.step: Step = "markers"
        self.error: str | None = None
        self.warning: str | None = None
        self.result: Calibration | None = None
        self._since: float | None = None  # when the current condition started holding
        self._capture_from: float | None = None
        self._frames: dict[str, list[HandObs]] = {"closed": [], "open": []}
        self._mic_start = 0
        self._noise: np.ndarray | None = None
        self._progress = 0.0

    @property
    def done(self) -> bool:
        return self.step == "done"

    @property
    def instruction(self) -> str:
        if self.markerless and self.step == "markers":
            return MARKERLESS_LOCK
        return INSTRUCTIONS[self.step]

    @property
    def progress(self) -> float:
        """0..1 through the current step."""
        return self._progress

    def update(self, t: float, pose: RacketPose, strumming: HandObs | None) -> None:
        if self.step == "markers":
            self._hold(t, pose.visible, MARKERS_HOLD_S, next_step="closed")
        elif self.step in ("closed", "open"):
            self._grip(t, strumming)
        elif self.step in ("quiet", "brush"):
            self._listen(t)

    # -- steps --------------------------------------------------------------------------

    def _hold(self, t: float, ok: bool, need: float, next_step: Step) -> None:
        if not ok:
            self._since, self._progress = None, 0.0
            return
        if self._since is None:
            self._since = t
        self._progress = min(1.0, (t - self._since) / need)
        if t - self._since >= need:
            self._enter(next_step, t)

    def _grip(self, t: float, hand: HandObs | None) -> None:
        frames = self._frames[self.step]
        if hand is None:
            frames.clear()
            self._since = self._capture_from = None
            self._progress = 0.0
            return
        if self._since is None:
            self._since = t
        if t - self._since < SETTLE_S:
            self._progress = 0.0
            return
        if self._capture_from is None:
            self._capture_from = t
        frames.append(hand)
        self._progress = min(1.0, (t - self._capture_from) / GRIP_CAPTURE_S)
        if t - self._capture_from >= GRIP_CAPTURE_S:
            if self.step == "closed":
                self._enter("open", t)
            else:
                self._finish_grip(t)

    def _finish_grip(self, t: float) -> None:
        try:
            result = calibrate_grip(self._frames["closed"], self._frames["open"], self.base)
            c, o = result.grip_closed_score, result.grip_open_score
            assert c is not None and o is not None
            if o < MIN_GRIP_RATIO * c:
                raise ValueError(
                    f"pick grip ({c:.2f}) and open hand ({o:.2f}) look too alike: spread your"
                    " fingers wider"
                )
            self.result = result
        except ValueError as exc:
            self.error = f"{exc}; try again"
            self._frames = {"closed": [], "open": []}
            self._enter("closed", t)
            return
        self.error = None
        self._enter("quiet" if self.mic is not None else "done", t)

    def _listen(self, t: float) -> None:
        assert self.mic is not None
        if self._since is None:
            self._since, self._mic_start = t, self.mic.written
        need = QUIET_S if self.step == "quiet" else BRUSH_S
        self._progress = min(1.0, (t - self._since) / need)
        if t - self._since < need:
            return
        audio, _ = self.mic.read_since(self._mic_start)
        if self.step == "quiet":
            self._noise = audio
            self._enter("brush", t)
            return
        assert self._noise is not None and self.result is not None
        self.result = calibrate_mic(self._noise, audio, self.cfg.mic, self.result)
        margin = mic_margin(self.result, self.cfg.mic)
        if margin is None:
            self.warning = "No brushes heard: move the mic closer or use headphones"
        elif margin < LOW_MIC_MARGIN:
            self.warning = f"Brushes are only {margin:.1f}x the threshold: mic gate may miss"
        self._enter("done", t)

    def _enter(self, step: Step, t: float) -> None:
        self.step = step
        self._since = self._capture_from = None
        self._progress = 0.0
        if step == "done" and self.result is None:
            self.result = replace(self.base)
