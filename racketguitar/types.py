"""Shared contracts between work packages. Frozen after WP0.

Every package may import this module and ``racketguitar.config``, nothing else outside
its own work package. Changing anything here needs the lead's sign-off, because all
four parallel work packages build against it.

Conventions every package relies on:

* **Time.** ``t`` fields are ``time.monotonic()`` seconds. For camera-derived data it
  is the frame's capture time, not the time processing finished.
* **Pixels.** Pixel coordinates are (x, y) in the full-resolution, un-mirrored camera
  frame, x to the right and y down. Detectors that work on a downscaled copy (e.g. the
  640x360 hand-landmarker input) scale results back before emitting. Mirroring is a
  display-only concern of the overlay.
* **Racket frame.** ``u`` points from the throat marker centre toward the tip marker
  centre. ``v`` is ``u`` rotated by +90 degrees in image coordinates:
  ``v = (-u.y, u.x)``. Racket-frame coordinates (u, v) are measured in head lengths
  from ``origin``, so ``px = origin + (u * U + v * V) * head_len``.
* **Strings.** Index 0 is low E, 5 is high E. "down" means the fingertip moves from
  the low-E side toward the high-E side, as on a real guitar.
* **MIDI.** Note numbers with middle C = 60. Standard tuning is 40 45 50 55 59 64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Vec2",
    "RacketPose",
    "HandObs",
    "Onset",
    "StringHit",
    "NoteEvent",
    "Direction",
    "NoteKind",
    "GripState",
    "StrumGate",
    "NUM_STRINGS",
    "NUM_LANDMARKS",
]

NUM_STRINGS = 6
NUM_LANDMARKS = 21  # MediaPipe hand landmark count

Direction = Literal["down", "up"]
NoteKind = Literal["on", "damp", "all_off"]
GripState = Literal["closed", "open", "unknown"]


class Vec2(NamedTuple):
    """A 2D point or vector in pixels. A tuple, so ``np.asarray(vec)`` works."""

    x: float
    y: float


@dataclass(frozen=True)
class RacketPose:  # from WP1
    t: float  # capture time, time.monotonic() seconds
    visible: bool
    origin: Vec2  # throat marker centre, pixels
    u: Vec2  # unit vector throat -> tip, pixels
    v: Vec2  # unit perpendicular
    head_len: float  # pixels, throat -> tip


@dataclass(frozen=True)
class HandObs:  # from WP1
    """One detected hand in one frame, in MediaPipe landmark order.

    Holds numpy arrays, so ``==`` and ``hash()`` on whole instances are not
    meaningful; compare fields with ``np.array_equal`` instead.
    """

    t: float
    landmarks: np.ndarray  # (21, 2) pixels, filtered
    raw: np.ndarray  # (21, 2) pixels, unfiltered


@dataclass(frozen=True)
class Onset:  # from WP4
    t: float
    strength: float  # how far the onset rose above the adaptive noise floor (ratio, >= 1)


@dataclass(frozen=True)
class StringHit:  # inside WP2
    t: float
    string: int  # 0 = low E ... 5 = high E
    direction: Literal["down", "up"]
    speed: float  # head lengths per second, perpendicular

    def __post_init__(self) -> None:
        if not 0 <= self.string < NUM_STRINGS:
            raise ValueError(f"string out of range: {self.string}")
        if self.direction not in ("down", "up"):
            raise ValueError(f"bad direction: {self.direction!r}")


@dataclass(frozen=True)
class NoteEvent:  # WP2 -> WP3
    """A command for the synth.

    * ``on``: pluck ``string`` at ``midi`` with ``velocity``; restarts that string's voice.
    * ``damp``: stop ``string`` ringing; ``midi`` is None.
    * ``all_off``: stop every string; ``string`` is -1 and ``midi`` is None.

    ``velocity`` must still be 1..127 for ``damp`` and ``all_off``; the synth ignores it.
    Strings the current chord leaves unplayed produce no event at all.
    """

    t_detect: float
    kind: Literal["on", "damp", "all_off"]
    string: int
    midi: int | None
    velocity: int  # 1..127

    def __post_init__(self) -> None:
        if not 1 <= self.velocity <= 127:
            raise ValueError(f"velocity out of range: {self.velocity}")
        if self.kind == "on":
            if not 0 <= self.string < NUM_STRINGS or self.midi is None:
                raise ValueError("'on' needs a string 0..5 and a midi note")
            if not 0 <= self.midi <= 127:
                raise ValueError(f"midi out of range: {self.midi}")
        elif self.kind == "damp":
            if not 0 <= self.string < NUM_STRINGS or self.midi is not None:
                raise ValueError("'damp' needs a string 0..5 and midi=None")
        elif self.kind == "all_off":
            if self.string != -1 or self.midi is not None:
                raise ValueError("'all_off' needs string=-1 and midi=None")
        else:
            raise ValueError(f"bad kind: {self.kind!r}")


@runtime_checkable
class StrumGate(Protocol):
    """Decides whether a string crossing is allowed to sound.

    Implemented by WP4 (``play/gate.py``): pick-grip check by default, optional mic
    onset window. Consumed by WP2 (``play/engine.py``), which receives an instance by
    injection and never imports ``play/gate.py``.

    Not thread-safe; every call comes from the vision thread. Per frame the integrator
    first drains the mic's onset queue into ``observe_onset``, then the engine calls
    ``observe_hand`` once with the frame time, then ``allows`` (and ``note_emitted`` for
    each note it plays) with each hit's interpolated crossing time.

    Ordering: ``observe_hand`` times are non-decreasing across frames, and so are onset
    times. A hit's ``t`` can be up to one frame *earlier* than the same frame's
    ``observe_hand`` time, and an onset can be later than a following ``allows(t)``.

    ``allows`` is not a pure query: a passing crossing can anchor the mic gate's sweep
    hold. Call it exactly once per detected crossing, never speculatively.
    """

    def observe_hand(self, t: float, hand: HandObs | None) -> None:
        """Called once per frame with the strumming hand, or None when it is not seen."""
        ...

    def observe_onset(self, onset: Onset) -> None:
        """Called for every mic onset (a no-op when the mic gate is off)."""
        ...

    def note_emitted(self, t: float) -> None:
        """The engine emitted a note-on detected at ``t``; used to ignore speaker bleed."""
        ...

    def allows(self, t: float) -> bool:
        """Whether a crossing detected at ``t`` may play."""
        ...

    @property
    def grip(self) -> GripState:
        """Latest classified strumming-hand grip, for the overlay."""
        ...
