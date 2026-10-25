"""The per-frame play step, shared by the live app and the replay harness.

Live and replay must run exactly the same code from the moment a frame's observations
exist, or a replay would prove nothing. ``Pipeline.step`` is that code: it applies any
song-mode chord relabel, feeds the frame's drained mic onsets to the gate, then runs
the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from racketguitar.config import Calibration, Config
from racketguitar.play.engine import Engine, EngineSnapshot
from racketguitar.play.gate import Gate, make_gate
from racketguitar.types import HandObs, NoteEvent, Onset, RacketPose

__all__ = ["FrameInput", "Pipeline"]


@dataclass(frozen=True, eq=False)
class FrameInput:
    """Everything the play logic saw for one camera frame."""

    t: float  # capture time
    proc_t: float  # monotonic time processing began, i.e. when onsets were drained
    pose: RacketPose
    hands: tuple[HandObs, ...]
    onsets: tuple[Onset, ...]  # drained just before this frame, oldest first
    # Song mode: relabel the handle sections from this frame on (None = unchanged).
    chords: tuple[tuple[str, tuple[int | None, ...]], ...] | None = None


class Pipeline:
    """Gate plus engine for one config and calibration. Vision thread only."""

    def __init__(self, cfg: Config, calibration: Calibration | None = None) -> None:
        self.cfg = cfg
        self.calibration = calibration or Calibration()
        self.gate: Gate = make_gate(cfg, self.calibration)
        self.engine = Engine(cfg, self.gate)

    def step(self, frame: FrameInput) -> list[NoteEvent]:
        if frame.chords is not None:
            self.engine.set_chords(frame.chords)
        for onset in frame.onsets:
            self.gate.observe_onset(onset)
        return self.engine.process(frame.pose, frame.hands)

    def run(self, frames: Sequence[FrameInput]) -> list[list[NoteEvent]]:
        """Step through ``frames`` in order; the events of each frame."""
        return [self.step(f) for f in frames]

    def snapshot(self) -> EngineSnapshot:
        return self.engine.snapshot()
