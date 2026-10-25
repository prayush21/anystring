"""Plays a song sheet as a timeline: which chords sit on the handle, and when.

The handle has a fixed number of sections (``len(ChordConfig.chords)``). A song with
more chords than that has its sections relabelled as it goes. Which section gets which
chord is planned once, up front, by furthest-next-use replacement (Belady): a section
whose chord is needed again latest (ties: the one free soonest) is given the new chord,
and it is relabelled as
soon as its old chord was last needed, so the player sees each new chord on the handle
as early as possible. A chord already on the handle never moves.

``SongPlayer.state(t)`` is a pure function of the capture time ``t`` (after
``start``), so the overlay and the play step agree frame by frame and tests need no
clock. Chords still come only from which section the fretting hand grips; the player
only decides the labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from racketguitar.config import Config
from racketguitar.song.sheet import Song
from racketguitar.song.voicings import parse_chord, voicing

__all__ = [
    "TimedChord",
    "TimedLine",
    "Timeline",
    "SlotChange",
    "SongState",
    "SongPlayer",
    "build_timeline",
    "plan_slots",
]

Phase = Literal["ready", "countin", "playing", "done"]
ChordSet = tuple[tuple[str, tuple[int | None, ...]], ...]


@dataclass(frozen=True)
class TimedChord:
    name: str
    beat: float  # start, in beats from the first chord of the song
    beats: float
    line: int  # index into Timeline.lines

    @property
    def end(self) -> float:
        return self.beat + self.beats


@dataclass(frozen=True)
class TimedLine:
    section: str
    text: str
    beat: float
    beats: float
    labels: tuple[tuple[int, str], ...]  # (column, chord name) to draw over the text


@dataclass(frozen=True)
class Timeline:
    lines: tuple[TimedLine, ...]
    chords: tuple[TimedChord, ...]  # consecutive repeats merged
    beats_per_bar: int

    @property
    def total_beats(self) -> float:
        return max(self.lines[-1].beat + self.lines[-1].beats, self.chords[-1].end)


@dataclass(frozen=True)
class SlotChange:
    beat: float  # from this beat on (-inf = before the song starts)
    slots: tuple[str, ...]  # chord name per handle section, head -> butt


@dataclass(frozen=True)
class SongState:
    """What the overlay shows and the play step uses, for one frame."""

    phase: Phase
    beat: float  # song beat (negative during the count-in)
    count: int | None  # count-in number, 0 = "GO"; None outside the count-in
    line: int | None  # index of the current lyric line
    chord: str | None  # chord the song is on now
    next_chord: str | None
    next_slot: int | None  # handle section of the next chord
    beats_to_next: float | None
    slots: tuple[str, ...]
    chords: ChordSet  # what the handle sections play now


def build_timeline(song: Song, beats_per_chord: float, simplify_slash: bool) -> Timeline:
    """Lay the sheet's lines end to end. ``beats_per_chord`` is the default length."""
    bpb = song.beats_per_bar
    lines: list[TimedLine] = []
    chords: list[TimedChord] = []
    beat = 0.0

    def name_of(raw: str) -> str:
        return parse_chord(raw).base_name if simplify_slash else raw

    for line in song.lines:
        start = beat
        labels: list[tuple[int, str]] = []
        for span in line.chords:
            name = name_of(span.name)
            beats = span.bars * bpb if span.bars is not None else beats_per_chord
            if chords and chords[-1].name == name and chords[-1].end == beat:
                last = chords[-1]
                chords[-1] = TimedChord(last.name, last.beat, last.beats + beats, last.line)
            else:
                chords.append(TimedChord(name, beat, beats, len(lines)))
            if not labels or labels[-1][1] != name:
                labels.append((span.col, name))
            beat += beats
        if not line.chords:  # a lyric with no chord over it holds the last chord
            beats = beats_per_chord
            if chords:
                last = chords[-1]
                chords[-1] = TimedChord(last.name, last.beat, last.beats + beats, last.line)
            beat += beats
        lines.append(TimedLine(line.section, line.text, start, beat - start, tuple(labels)))
    if not chords:
        raise ValueError("the song has no chords")
    return Timeline(tuple(lines), tuple(chords), bpb)


def plan_slots(chords: tuple[TimedChord, ...], sections: int) -> tuple[SlotChange, ...]:
    """Handle labels over the song; see the module docstring."""
    distinct = list(dict.fromkeys(c.name for c in chords))
    n = min(sections, len(distinct))
    slots = distinct[:n]
    changes = [SlotChange(-math.inf, tuple(slots))]
    for i, c in enumerate(chords):
        if c.name in slots:
            continue

        def next_use(name: str, i: int = i) -> float:
            return next((j for j in range(i + 1, len(chords)) if chords[j].name == name), math.inf)

        def freed(name: str, i: int = i) -> float:
            return max((p.end for p in chords[:i] if p.name == name), default=-math.inf)

        # Furthest next use; among ties, the chord freed earliest (earliest relabel).
        k = max(range(n), key=lambda s: (next_use(slots[s]), -freed(slots[s])))
        free_at = freed(slots[k])
        when = max(free_at, changes[-1].beat)
        slots[k] = c.name
        if when == changes[-1].beat:
            changes[-1] = SlotChange(when, tuple(slots))
        else:
            changes.append(SlotChange(when, tuple(slots)))
    return tuple(changes)


class SongPlayer:
    """Ready -> count-in -> playing -> done, for one song sheet."""

    def __init__(self, song: Song, cfg: Config) -> None:
        sc = cfg.song
        self.song = song
        self.speed = sc.speed
        self.bpm = song.tempo * sc.speed
        self.count_in = max(1, sc.count_in_beats)
        bpb = song.beats_per_bar
        bars = song.chord_bars if song.chord_bars is not None else sc.chord_bars
        self.timeline = build_timeline(song, bars * bpb, sc.simplify_slash)
        self.plan = plan_slots(self.timeline.chords, len(cfg.chords.chords))
        tuning = cfg.strings.tuning
        self.voicings = {c.name: voicing(c.name, tuning, song.capo) for c in self.timeline.chords}
        self._t0: float | None = None  # time of the first count-in beat

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    @property
    def started(self) -> bool:
        return self._t0 is not None

    @property
    def start_time(self) -> float | None:
        """Capture time of the first count-in beat, None before ``start``."""
        return self._t0

    def start(self, t: float) -> None:
        self._t0 = t

    def stop(self) -> None:
        self._t0 = None

    def slots_at(self, beat: float) -> tuple[str, ...]:
        slots = self.plan[0].slots
        for change in self.plan:
            if change.beat > beat:
                break
            slots = change.slots
        return slots

    def chord_set(self, slots: tuple[str, ...]) -> ChordSet:
        return tuple((name, self.voicings[name]) for name in slots)

    def state(self, t: float) -> SongState:
        chords = self.timeline.chords
        if self._t0 is None:
            beat, phase, count = -math.inf, "ready", None
        else:
            beat = (t - self._t0) / self.seconds_per_beat - self.count_in
            if beat < 0:
                phase = "countin"
                count = max(0, int(-math.floor(beat)) - 1)
            elif beat >= self.timeline.total_beats:
                phase, count = "done", None
            else:
                phase, count = "playing", None
        i = self._chord_index(beat)
        current = chords[i] if i is not None and phase == "playing" else None
        nxt_i = 0 if i is None else i + 1
        nxt = chords[nxt_i] if phase != "done" and nxt_i < len(chords) else None
        slots = self.slots_at(nxt.beat if current is None and nxt is not None else beat)
        next_slot = None
        if nxt is not None:
            upcoming = self.slots_at(nxt.beat)
            next_slot = upcoming.index(nxt.name)
        line = None
        if phase in ("playing", "done"):
            line = self._line_index(min(beat, self.timeline.total_beats - 1e-9))
        return SongState(
            phase=phase,
            beat=beat,
            count=count,
            line=line,
            chord=current.name if current is not None else None,
            next_chord=nxt.name if nxt is not None else None,
            next_slot=next_slot,
            beats_to_next=None if nxt is None or beat == -math.inf else nxt.beat - beat,
            slots=slots,
            chords=self.chord_set(slots),
        )

    def _chord_index(self, beat: float) -> int | None:
        idx = None
        for i, c in enumerate(self.timeline.chords):
            if c.beat <= beat:
                idx = i
            else:
                break
        return idx

    def _line_index(self, beat: float) -> int:
        idx = 0
        for i, line in enumerate(self.timeline.lines):
            if line.beat <= beat:
                idx = i
            else:
                break
        return idx
