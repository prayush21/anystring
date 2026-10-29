"""What a take is judged from: per-frame play state, string hits and the song's chords.

These are plain values, copied from the engine's snapshots by the integration layer
(live, or by replaying a recorded session), so the analysis needs no engine, gate or
song player and tests can write a take by hand. Times are capture times, as everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from anystring.types import Direction, GripState

__all__ = ["HitOutcome", "HitSample", "FrameSample", "DueChord", "SongInfo", "TakeLog"]

# played: it sounded. gated: the strum gate blocked it (pick grip not held). silent: the
# chord leaves that string unplayed, so nothing was meant to sound.
HitOutcome = Literal["played", "gated", "silent"]


@dataclass(frozen=True, slots=True)
class HitSample:
    t: float
    string: int  # 0 = low E
    direction: Direction
    speed: float  # head lengths/s, perpendicular
    velocity: int
    outcome: HitOutcome
    chord: str | None  # chord the handle gave, None = open strings


@dataclass(frozen=True, slots=True)
class FrameSample:
    t: float
    visible: bool  # racket tracked
    chord: str | None  # chord the fretting hand holds, None = open strings
    grip: GripState  # gate's grip classification of the strumming hand
    muted: bool  # palm mute fired and the palm still rests
    strumming: bool  # a strumming hand was seen
    refractory_blocked: int  # running count of double crossings the engine dropped


@dataclass(frozen=True, slots=True)
class DueChord:
    name: str
    t: float  # capture time the chord is due
    end: float  # when the next chord is due (or the song ends)


@dataclass(frozen=True)
class SongInfo:
    title: str
    bpm: float  # as played (sheet tempo times the speed setting)
    beats_per_bar: int
    t_beat0: float  # capture time of the song's first beat, after the count-in
    chords: tuple[DueChord, ...]
    finished: bool  # played to the end rather than stopped

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm


@dataclass(frozen=True)
class TakeLog:
    frames: tuple[FrameSample, ...]  # oldest first
    hits: tuple[HitSample, ...]  # oldest first
    song: SongInfo | None = None

    @property
    def seconds(self) -> float:
        return self.frames[-1].t - self.frames[0].t if len(self.frames) > 1 else 0.0
