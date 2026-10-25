"""The coach's view of play: engine snapshots copied into a ``coach.log.TakeLog``.

``TakeLogger.observe`` runs once per frame on the vision thread and only appends a few
small values, so it costs the play path nothing measurable. The app keeps the last
``CoachConfig.history_s`` seconds; ``log`` cuts a take out of them. ``log_from_session``
builds the same log from a recording by stepping it through a fresh ``Pipeline``, which
gives exactly what the live app saw.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from anystring.coach.log import DueChord, FrameSample, HitSample, SongInfo, TakeLog
from anystring.pipeline import Pipeline
from anystring.play.engine import EngineSnapshot, HitRecord
from anystring.replay.record import load_session
from anystring.song.player import SongPlayer
from anystring.song.sheet import load_sheet

__all__ = ["TakeLogger", "song_info", "log_from_session"]


class TakeLogger:
    """Rolling record of the play state, for feedback. Vision thread only."""

    def __init__(self, history_s: float) -> None:
        self.history_s = history_s
        self._frames: deque[FrameSample] = deque()
        self._hits: deque[HitSample] = deque()
        self._last: HitRecord | None = None  # newest hit already copied

    @property
    def last_hit_t(self) -> float | None:
        return self._hits[-1].t if self._hits else None

    def observe(self, snap: EngineSnapshot) -> None:
        recent = snap.recent
        if recent and recent[-1] is not self._last:
            # New hits follow the last one copied; if it is gone (a rebuilt engine
            # starts a new list), every hit in the list is new.
            start = 0
            for i in range(len(recent) - 1, -1, -1):
                if recent[i] is self._last:
                    start = i + 1
                    break
            for r in recent[start:]:
                outcome = "played" if r.played else "silent" if r.midi is None else "gated"
                h = r.hit
                self._hits.append(
                    HitSample(h.t, h.string, h.direction, h.speed, r.velocity, outcome, snap.chord)
                )
            self._last = recent[-1]
        self._frames.append(
            FrameSample(
                snap.t,
                snap.visible,
                snap.chord,
                snap.grip,
                snap.muted,
                snap.strumming is not None,
                snap.refractory_blocked,
            )
        )
        old = snap.t - self.history_s
        while self._frames and self._frames[0].t < old:
            self._frames.popleft()
        while self._hits and self._hits[0].t < old:
            self._hits.popleft()

    def log(self, start: float, end: float | None = None, song: SongInfo | None = None) -> TakeLog:
        """The take from ``start`` to ``end`` (default: the latest frame)."""
        end = float("inf") if end is None else end
        return TakeLog(
            tuple(f for f in self._frames if start <= f.t <= end),
            tuple(h for h in self._hits if start <= h.t <= end),
            song,
        )

    def free_take_start(self, now: float, window_s: float, gap_s: float) -> float:
        """Start of the latest stretch of playing: after the last pause of ``gap_s``
        without a string crossing, and at most ``window_s`` before ``now``."""
        start = now - window_s
        times = [h.t for h in self._hits if h.t >= start]
        for a, b in zip(reversed(times[:-1]), reversed(times[1:]), strict=True):
            if b - a >= gap_s:
                return max(start, b - 0.5)
        return start


def song_info(player: SongPlayer, finished: bool) -> SongInfo:
    """A started song's chords as capture times."""
    assert player.start_time is not None
    spb = player.seconds_per_beat
    beat0 = player.start_time + player.count_in * spb
    chords = tuple(
        DueChord(c.name, beat0 + c.beat * spb, beat0 + c.end * spb) for c in player.timeline.chords
    )
    return SongInfo(
        player.song.title, player.bpm, player.timeline.beats_per_bar, beat0, chords, finished
    )


def log_from_session(path: str | Path) -> TakeLog:
    """The take log of a recorded session, with its last song if one was played and its
    sheet is still where it was."""
    session = load_session(path)
    pipeline = Pipeline(session.config, session.calibration)
    logger = TakeLogger(float("inf"))
    for fr in session.frames:
        pipeline.step(fr.inp)
        logger.observe(pipeline.snapshot())
    if not session.frames:
        return logger.log(0.0)
    first, last = session.frames[0].inp.t, session.frames[-1].inp.t
    runs = [r for r in session.meta.get("songs", []) if r.get("start_t", -1) <= last]
    sheet = Path(runs[-1]["path"]) if runs and runs[-1].get("path") else None
    if sheet is None or not sheet.exists():
        return logger.log(first)
    player = SongPlayer(load_sheet(sheet), session.config)
    player.start(float(runs[-1]["start_t"]))
    info = song_info(player, finished=player.state(last).phase == "done")
    return logger.log(max(first, player.start_time), song=info)
