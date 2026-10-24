"""The play engine: one call per camera frame turns a pose and hands into note events.

Per frame (vision thread):

1. Build the racket frame from the pose and resolve the string orientation.
2. Split the hands: the fretting hand is the one whose knuckle (``ChordConfig.
   knuckle_landmark``) is nearest the handle, if it is on the handle at all; the
   strumming hand is the remaining hand nearest the head.
3. ``gate.observe_hand(pose.t, strumming hand or None)``, exactly once.
4. Chord section from the fretting hand; palm mute from the strumming hand. A mute
   emits ``damp`` for every string that has sounded since the last mute (not
   ``all_off``, which is left for a stop/panic action).
5. String hits from the strumming fingertips. For each hit, in time order:
   ``gate.allows(hit.t)``, then the chord's note for that string (unplayed strings emit
   nothing), velocity from speed, a ``NoteEvent("on")`` and ``gate.note_emitted(hit.t)``.

``snapshot()`` returns an immutable view of the latest frame for the overlay; it is
safe to read from another thread.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from racketguitar.config import ChordConfig, Config, VelocityConfig
from racketguitar.geometry.racket import (
    HandleAxis,
    RacketFrame,
    StringLayout,
    StringOrientation,
)
from racketguitar.play.chords import ChordSelector
from racketguitar.play.mute import PalmMute, palm_centre
from racketguitar.play.strum import StrumDetector
from racketguitar.types import GripState, HandObs, NoteEvent, RacketPose, StringHit, StrumGate

__all__ = [
    "RECENT_HITS",
    "DAMP_VELOCITY",
    "velocity_for_speed",
    "split_hands",
    "HitRecord",
    "EngineSnapshot",
    "Engine",
]

RECENT_HITS = 32  # hits kept in the snapshot for the overlay
DAMP_VELOCITY = 1  # damp events still need a valid velocity; the synth ignores it


def velocity_for_speed(speed: float, cfg: VelocityConfig) -> int:
    """Perpendicular speed (head lengths/s) -> MIDI velocity on a gamma curve."""
    span = cfg.speed_at_max - cfg.speed_at_min
    x = (speed - cfg.speed_at_min) / span if span > 0 else 1.0
    x = min(1.0, max(0.0, x))
    vel = cfg.min_velocity + (cfg.max_velocity - cfg.min_velocity) * x**cfg.gamma
    return int(min(127, max(1, round(vel))))


def split_hands(
    frame: RacketFrame,
    hands: Sequence[HandObs],
    handle: HandleAxis,
    knuckle: int,
    *,
    max_v: float = ChordConfig.fret_max_v,
    t_margin: float = ChordConfig.fret_t_margin,
) -> tuple[HandObs | None, HandObs | None]:
    """``(fretting, strumming)`` for one frame; either may be None.

    A hand can be the fretting hand only if its knuckle is on the handle: within
    ``max_v`` of the axis (head lengths) and at most ``t_margin`` past either end.
    """
    fret_i, best = -1, np.inf
    for i, h in enumerate(hands):
        uv = frame.to_uv(h.landmarks[knuckle])
        t = handle.t_of(uv)
        if -t_margin <= t <= 1.0 + t_margin and abs(uv[1]) <= max_v:
            d = handle.distance(uv)
            if d < best:
                fret_i, best = i, d
    strum_i, best = -1, np.inf
    for i, h in enumerate(hands):
        if i == fret_i:
            continue
        uv = frame.to_uv(palm_centre(h.landmarks))
        d = float(np.hypot(uv[0] - 0.5, uv[1]))
        if d < best:
            strum_i, best = i, d
    return (
        hands[fret_i] if fret_i >= 0 else None,
        hands[strum_i] if strum_i >= 0 else None,
    )


@dataclass(frozen=True, eq=False)
class HitRecord:
    """A detected crossing and what became of it."""

    hit: StringHit
    velocity: int
    midi: int | None  # None when the chord leaves this string unplayed
    played: bool  # False when the gate blocked it or the string is unplayed


@dataclass(frozen=True, eq=False)
class EngineSnapshot:
    """Read-only state of the latest frame, for the overlay."""

    t: float
    visible: bool  # racket visible this frame
    frame: RacketFrame | None
    layout: StringLayout
    handle: HandleAxis
    low_e_neg_v: bool
    chord_names: tuple[str, ...]  # head -> butt
    zone: int | None  # active handle section, None = open strings
    chord: str | None  # active chord name, None = open strings
    voicing: tuple[int | None, ...]  # MIDI per string, None = unplayed
    muted: bool  # palm mute fired and the palm is still resting
    grip: GripState  # gate's latest classification
    strumming: HandObs | None
    fretting: HandObs | None
    fingertips_uv: np.ndarray | None  # (k, 2) strumming fingertips, racket frame
    recent: tuple[HitRecord, ...]  # oldest first, at most RECENT_HITS

    def string_v(self) -> np.ndarray:
        """v of each string, indexed by string (0 = low E)."""
        return self.layout.string_v(self.low_e_neg_v)

    def strings_px(self) -> np.ndarray | None:
        """(6, 2, 2) string segment endpoints in pixels, by string; None if not visible."""
        return None if self.frame is None else self.layout.segments_px(self.frame, self.low_e_neg_v)

    def handle_sections_px(self) -> np.ndarray | None:
        """(n, 2, 2) handle sections in pixels, head -> butt, matching ``chord_names``."""
        if self.frame is None:
            return None
        return self.handle.sections_px(self.frame, len(self.chord_names))

    def fingertips_px(self) -> np.ndarray | None:
        if self.frame is None or self.fingertips_uv is None:
            return None
        return self.frame.to_px(self.fingertips_uv)


class Engine:
    """Vision-thread entry point. Receives its ``StrumGate`` by injection."""

    def __init__(self, cfg: Config, gate: StrumGate) -> None:
        self._cfg = cfg
        self._gate = gate
        self._handle = HandleAxis.from_config(cfg.chords)
        self._knuckle = cfg.chords.knuckle_landmark
        self._orientation = StringOrientation(cfg.strings.low_e_side, cfg.strings.auto_side_margin)
        self._strum = StrumDetector(cfg.strings, cfg.strum, cfg.hands.skip_first_frames_back)
        self._chords = ChordSelector(cfg.chords, cfg.strings)
        self._mute = PalmMute(cfg.mute, cfg.strings.head_half_width)
        self._ringing = [False] * cfg.strings.count
        self._recent: deque[HitRecord] = deque(maxlen=RECENT_HITS)
        self._recent_tuple: tuple[HitRecord, ...] = ()
        self._snapshot = self._make_snapshot(0.0, None, None, None)

    def snapshot(self) -> EngineSnapshot:
        return self._snapshot

    def reset(self) -> None:
        """Forget all hand, chord and mute state (e.g. between replay sessions)."""
        self._strum.reset()
        self._chords.reset()
        self._mute.reset()
        self._ringing = [False] * self._cfg.strings.count

    def process(self, pose: RacketPose, hands: Sequence[HandObs]) -> list[NoteEvent]:
        """Run one frame. ``pose.t`` is the frame's capture time."""
        t = pose.t
        frame = RacketFrame.from_pose(pose) if pose.visible else None
        fretting = strumming = None
        if frame is not None:
            low_e_neg_v = self._orientation.update(frame)
            fretting, strumming = split_hands(
                frame,
                hands,
                self._handle,
                self._knuckle,
                max_v=self._cfg.chords.fret_max_v,
                t_margin=self._cfg.chords.fret_t_margin,
            )
        else:
            low_e_neg_v = self._orientation.low_e_neg_v

        self._gate.observe_hand(t, strumming)

        if frame is not None:
            handle_t = (
                self._handle.t_of(frame.to_uv(fretting.landmarks[self._knuckle]))
                if fretting is not None
                else None
            )
            self._chords.update(t, handle_t)

        events: list[NoteEvent] = []
        if self._mute.update(frame, strumming):
            for s, ringing in enumerate(self._ringing):
                if ringing:
                    events.append(NoteEvent(t, "damp", s, None, DAMP_VELOCITY))
            self._ringing = [False] * len(self._ringing)

        hits = self._strum.update(frame, strumming, low_e_neg_v)
        if hits:
            voicing = self._chords.voicing
            for hit in hits:
                allowed = self._gate.allows(hit.t)
                midi = voicing[hit.string]
                vel = velocity_for_speed(hit.speed, self._cfg.velocity)
                played = allowed and midi is not None
                if played:
                    events.append(NoteEvent(hit.t, "on", hit.string, midi, vel))
                    self._gate.note_emitted(hit.t)
                    self._ringing[hit.string] = True
                self._recent.append(HitRecord(hit, vel, midi, played))
            self._recent_tuple = tuple(self._recent)

        self._snapshot = self._make_snapshot(t, frame, fretting, strumming)
        return events

    def _make_snapshot(
        self,
        t: float,
        frame: RacketFrame | None,
        fretting: HandObs | None,
        strumming: HandObs | None,
    ) -> EngineSnapshot:
        return EngineSnapshot(
            t=t,
            visible=frame is not None,
            frame=frame,
            layout=self._strum.layout,
            handle=self._handle,
            low_e_neg_v=self._orientation.low_e_neg_v,
            chord_names=self._chords.names,
            zone=self._chords.zone,
            chord=self._chords.chord,
            voicing=self._chords.voicing,
            muted=self._mute.muted,
            grip=self._gate.grip,
            strumming=strumming,
            fretting=fretting,
            fingertips_uv=self._strum.tips_uv,
            recent=self._recent_tuple,
        )
