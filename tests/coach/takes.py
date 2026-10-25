"""Hand-written takes for the coach tests: a song's chord timeline, the chord the hand
holds frame by frame, and strokes as runs of string hits."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from anystring.coach.log import DueChord, FrameSample, HitSample, SongInfo, TakeLog

FPS = 60.0
T0 = 10.0  # beat 0


def song(names: Iterable[str], beats: float = 4.0, bpm: float = 120.0) -> SongInfo:
    spb = 60.0 / bpm
    chords, t = [], T0
    for name in names:
        chords.append(DueChord(name, t, t + beats * spb))
        t += beats * spb
    return SongInfo("Test Song", bpm, 4, T0, tuple(chords), finished=True)


def held_as_due(info: SongInfo, delays: dict[str, float] | None = None) -> Callable:
    """The chord the hand holds: the due one, reached ``delays[name]`` s late."""
    delays = delays or {}

    def chord_at(t: float) -> str | None:
        held = None
        for c in info.chords:
            if t >= c.t + delays.get(c.name, 0.0):
                held = c.name
        return held

    return chord_at


def frames(
    t0: float,
    t1: float,
    chord_at: Callable[[float], str | None] = lambda t: None,
    *,
    muted_at: Callable[[float], bool] = lambda t: False,
    blocked_at: Callable[[float], int] = lambda t: 0,
    visible_at: Callable[[float], bool] = lambda t: True,
) -> tuple[FrameSample, ...]:
    out = []
    k = 0
    while (t := t0 + k / FPS) <= t1:
        out.append(
            FrameSample(t, visible_at(t), chord_at(t), "closed", muted_at(t), True, blocked_at(t))
        )
        k += 1
    return tuple(out)


def stroke(
    t: float,
    direction: str = "down",
    *,
    strings: int = 6,
    gated: int = 0,
    chord: str | None = "C",
    speed: float = 3.0,
) -> list[HitSample]:
    """``strings`` crossings 8 ms apart; the last ``gated`` were blocked by the gate."""
    order = range(strings) if direction == "down" else range(5, 5 - strings, -1)
    return [
        HitSample(
            t + 0.008 * i,
            s,
            direction,
            speed,
            90,
            "gated" if i >= strings - gated else "played",
            chord,
        )
        for i, s in enumerate(order)
    ]


def take(
    hits: Iterable[HitSample], frames_: tuple[FrameSample, ...], info: SongInfo | None = None
) -> TakeLog:
    return TakeLog(frames_, tuple(sorted(hits, key=lambda h: h.t)), info)


def beat_strokes(info: SongInfo, offset: float = 0.0, chord_at=None) -> list[HitSample]:
    """Down on every beat and up on every off-beat, ``offset`` s late, over the song."""
    spb = info.seconds_per_beat
    chord_at = chord_at or held_as_due(info)
    hits: list[HitSample] = []
    n = int(round((info.chords[-1].end - info.t_beat0) / (spb / 2)))
    for k in range(n):
        t = info.t_beat0 + k * spb / 2 + offset
        hits += stroke(t, "down" if k % 2 == 0 else "up", chord=chord_at(t))
    return hits
