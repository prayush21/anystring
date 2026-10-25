"""Measure a take: timing, chord changes, strokes, grip, mutes; then rank what to say.

Everything here is computed from the take log, so it is exact and repeatable; the LLM
only puts the result into words. Conventions:

* A **stroke** is a run of string hits in one direction with less than
  ``CoachConfig.stroke_gap_ms`` between them. It is *missed* when the gate blocked every
  string it crossed, a *grip drop* when it blocked some after others sounded, and *thin*
  when it crossed fewer than ``thin_strings`` strings.
* **Timing** (song mode) is each sounding stroke's offset from the nearest point of a
  grid of ``beat_subdivision`` steps per beat; positive is late (dragging), negative is
  early (rushing). Outside a song there is no beat, so only the steadiness of the
  stroke rate is measured.
* A **chord change** from A to B is due at B's start. The hand *arrives* at the first
  frame it holds B for two frames running, looking from halfway through A; *lag* is the
  arrival time minus the due time. It *left early* when it let go of A more than
  ``change_early_ms`` before B was due. Strums between the due time and the arrival are
  on the wrong chord.
* A palm mute is *likely unintended* when strumming resumes within ``mute_resume_ms`` of
  it firing (and, in a song, before the song ends).
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import asdict, dataclass, replace

from anystring.coach.log import FrameSample, HitSample, SongInfo, TakeLog
from anystring.config import CoachConfig
from anystring.types import Direction

__all__ = [
    "Stroke",
    "StrokeStats",
    "TimingStats",
    "TempoStats",
    "ChangeEvent",
    "ChangeStats",
    "MuteStats",
    "Finding",
    "TakeReport",
    "strokes_of",
    "analyze",
    "facts",
]


@dataclass(frozen=True)
class Stroke:
    t: float  # first string crossing
    direction: Direction
    strings: int  # distinct strings crossed
    played: int
    gated: int
    speed: float  # mean crossing speed, head lengths/s
    chord: str | None

    @property
    def sounded(self) -> bool:
        return self.played > 0


@dataclass(frozen=True)
class StrokeStats:
    strokes: int
    down: int
    up: int
    missed: int  # the gate blocked the whole stroke
    grip_drops: int  # the gate blocked part of it
    thin: int
    down_strings: float  # mean strings crossed per stroke
    up_strings: float
    down_speed: float  # mean speed, head lengths/s
    up_speed: float
    double_hits_blocked: int


@dataclass(frozen=True)
class TimingStats:
    strokes: int
    mean_ms: float  # + late (dragging), - early (rushing)
    median_abs_ms: float
    on_time_share: float
    first_half_ms: float
    second_half_ms: float


@dataclass(frozen=True)
class TempoStats:
    start_per_min: float  # strokes a minute over the first third
    end_per_min: float  # and the last third
    unevenness: float  # spread of the gaps between strokes, as a share of their median


@dataclass(frozen=True)
class ChangeEvent:
    frm: str
    to: str
    due: float
    lag_ms: float | None  # None = never reached B while it was due
    left_early_ms: float | None  # how long before the due time A was let go, if early
    wrong_strums: int  # sounding strokes after the due time, before arriving


@dataclass(frozen=True)
class ChangeStats:
    frm: str
    to: str
    count: int
    late: int
    missed: int
    left_early: int
    mean_lag_ms: float | None  # over the changes that arrived
    worst_lag_ms: float | None
    mean_left_early_ms: float | None
    wrong_strums: int


@dataclass(frozen=True)
class MuteStats:
    mutes: int
    likely_unintended: int


@dataclass(frozen=True)
class Finding:
    key: str
    score: float  # how much it matters; higher first
    fact: str  # what was measured, in plain words
    tip: str  # what to do about it


@dataclass(frozen=True)
class TakeReport:
    seconds: float
    song: SongInfo | None
    strokes: StrokeStats
    timing: TimingStats | None  # song mode only
    tempo: TempoStats | None  # outside a song only
    changes: tuple[ChangeStats, ...]  # song mode only, slowest first
    chord_accuracy: float | None  # song mode: share of sounding strokes on the due chord
    mutes: MuteStats
    racket_lost_share: float
    enough: bool  # at least ``min_strokes`` strokes
    findings: tuple[Finding, ...]  # problems, most important first
    strengths: tuple[str, ...]


def strokes_of(hits: tuple[HitSample, ...], gap_ms: float) -> list[Stroke]:
    gap = gap_ms / 1000.0
    groups: list[list[HitSample]] = []
    for h in hits:
        if groups and h.direction == groups[-1][-1].direction and h.t - groups[-1][-1].t < gap:
            groups[-1].append(h)
        else:
            groups.append([h])
    return [_stroke(g) for g in groups]


def _stroke(hits: list[HitSample]) -> Stroke:
    return Stroke(
        t=hits[0].t,
        direction=hits[0].direction,
        strings=len({h.string for h in hits}),
        played=sum(h.outcome == "played" for h in hits),
        gated=sum(h.outcome == "gated" for h in hits),
        speed=statistics.fmean(h.speed for h in hits),
        chord=hits[0].chord,
    )


def analyze(log: TakeLog, cfg: CoachConfig) -> TakeReport:
    strokes = strokes_of(log.hits, cfg.stroke_gap_ms)
    stats = _stroke_stats(strokes, log.frames, cfg)
    song = log.song
    timing = _timing(strokes, song, cfg) if song is not None else None
    tempo = _tempo(strokes) if song is None else None
    events = _change_events(log, strokes, cfg) if song is not None else []
    changes = _change_stats(events, cfg)
    accuracy = _chord_accuracy(strokes, song) if song is not None else None
    mutes = _mutes(log.frames, strokes, song, cfg)
    lost = sum(not f.visible for f in log.frames) / len(log.frames) if log.frames else 0.0
    enough = len(strokes) >= cfg.min_strokes
    report = TakeReport(
        seconds=log.seconds,
        song=song,
        strokes=stats,
        timing=timing,
        tempo=tempo,
        changes=changes,
        chord_accuracy=accuracy,
        mutes=mutes,
        racket_lost_share=lost,
        enough=enough,
        findings=(),
        strengths=(),
    )
    if not enough:
        return report
    findings, strengths = _judge(report, cfg)
    return replace(report, findings=findings, strengths=strengths)


# -- measurements ---------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _stroke_stats(strokes: list[Stroke], frames: tuple[FrameSample, ...], cfg: CoachConfig):
    down = [s for s in strokes if s.direction == "down"]
    up = [s for s in strokes if s.direction == "up"]
    blocked = sum(
        max(0, b.refractory_blocked - a.refractory_blocked)
        for a, b in zip(frames, frames[1:], strict=False)
    )
    return StrokeStats(
        strokes=len(strokes),
        down=len(down),
        up=len(up),
        missed=sum(s.played == 0 and s.gated > 0 for s in strokes),
        grip_drops=sum(s.played > 0 and s.gated > 0 for s in strokes),
        thin=sum(s.strings < cfg.thin_strings for s in strokes),
        down_strings=_mean([s.strings for s in down]),
        up_strings=_mean([s.strings for s in up]),
        down_speed=_mean([s.speed for s in down]),
        up_speed=_mean([s.speed for s in up]),
        double_hits_blocked=blocked,
    )


def _in_song(t: float, song: SongInfo, margin: float = 0.0) -> bool:
    return song.chords[0].t - margin <= t < song.chords[-1].end


def _timing(strokes: list[Stroke], song: SongInfo, cfg: CoachConfig) -> TimingStats | None:
    step = song.seconds_per_beat / max(1, cfg.beat_subdivision)
    offsets = []
    for s in strokes:
        if s.sounded and _in_song(s.t, song, step / 2):
            k = round((s.t - song.t_beat0) / step)
            offsets.append((s.t - (song.t_beat0 + k * step)) * 1000.0)
    if not offsets:
        return None
    half = len(offsets) // 2
    return TimingStats(
        strokes=len(offsets),
        mean_ms=statistics.fmean(offsets),
        median_abs_ms=statistics.median(abs(o) for o in offsets),
        on_time_share=sum(abs(o) <= cfg.on_time_ms for o in offsets) / len(offsets),
        first_half_ms=_mean(offsets[:half]) if half else offsets[0],
        second_half_ms=_mean(offsets[half:]),
    )


def _tempo(strokes: list[Stroke]) -> TempoStats | None:
    times = [s.t for s in strokes]  # a stroke the gate blocked still keeps the rhythm
    gaps = [b - a for a, b in zip(times, times[1:], strict=False) if 0.1 <= b - a <= 2.0]
    if len(gaps) < 6:
        return None
    third = len(gaps) // 3
    median = statistics.median(gaps)
    spread = statistics.median(abs(g - median) for g in gaps) / median
    return TempoStats(
        start_per_min=60.0 / statistics.median(gaps[:third]),
        end_per_min=60.0 / statistics.median(gaps[-third:]),
        unevenness=spread,
    )


def _change_events(log: TakeLog, strokes: list[Stroke], cfg: CoachConfig) -> list[ChangeEvent]:
    assert log.song is not None
    ts = [f.t for f in log.frames]
    held = [f.chord for f in log.frames]
    events = []
    chords = log.song.chords
    for a, b in zip(chords, chords[1:], strict=False):
        if a.name == b.name:
            continue
        due = b.t
        if not ts or due > ts[-1]:
            break  # the take stopped before this change
        look = max(a.t, due - 0.5 * (due - a.t))
        i, end = bisect.bisect_left(ts, look), bisect.bisect_left(ts, b.end)
        arrival = None
        for j in range(i, end):
            if held[j] == b.name and (j + 1 >= len(held) or held[j + 1] == b.name):
                arrival = ts[j]
                break
        # Let go of A: the frame after the last one holding it, before arriving at B.
        stop = bisect.bisect_left(ts, arrival if arrival is not None else b.end)
        first = bisect.bisect_left(ts, a.t)
        leave = None
        for j in range(stop - 1, first - 1, -1):
            if held[j] == a.name:
                leave = ts[j + 1] if j + 1 < len(ts) else ts[j]
                break
        early = due - leave if leave is not None else None
        until = arrival if arrival is not None else b.end
        wrong = sum(s.sounded and due <= s.t < until and s.chord != b.name for s in strokes)
        events.append(
            ChangeEvent(
                a.name,
                b.name,
                due,
                None if arrival is None else (arrival - due) * 1000.0,
                early * 1000.0
                if early is not None and early * 1000.0 > cfg.change_early_ms
                else None,
                wrong,
            )
        )
    return events


def _change_stats(events: list[ChangeEvent], cfg: CoachConfig) -> tuple[ChangeStats, ...]:
    pairs: dict[tuple[str, str], list[ChangeEvent]] = {}
    for e in events:
        pairs.setdefault((e.frm, e.to), []).append(e)
    out = []
    for (frm, to), es in pairs.items():
        lags = [e.lag_ms for e in es if e.lag_ms is not None]
        early = [e.left_early_ms for e in es if e.left_early_ms is not None]
        out.append(
            ChangeStats(
                frm=frm,
                to=to,
                count=len(es),
                late=sum(lag > cfg.change_late_ms for lag in lags),
                missed=len(es) - len(lags),
                left_early=len(early),
                mean_lag_ms=statistics.fmean(lags) if lags else None,
                worst_lag_ms=max(lags) if lags else None,
                mean_left_early_ms=statistics.fmean(early) if early else None,
                wrong_strums=sum(e.wrong_strums for e in es),
            )
        )
    return tuple(sorted(out, key=_change_badness, reverse=True))


def _change_badness(c: ChangeStats) -> float:
    return (c.late + 2.0 * c.missed + 0.5 * c.left_early) / c.count + (c.mean_lag_ms or 0.0) / 1e4


def _due_at(song: SongInfo, t: float) -> str | None:
    i = bisect.bisect_right([c.t for c in song.chords], t) - 1
    return song.chords[i].name if 0 <= i and t < song.chords[i].end else None


def _chord_accuracy(strokes: list[Stroke], song: SongInfo) -> float | None:
    due = [(s, _due_at(song, s.t)) for s in strokes if s.sounded]
    due = [(s, name) for s, name in due if name is not None]
    if not due:
        return None
    return sum(s.chord == name for s, name in due) / len(due)


def _mutes(
    frames: tuple[FrameSample, ...], strokes: list[Stroke], song: SongInfo | None, cfg: CoachConfig
) -> MuteStats:
    fired = [b.t for a, b in zip(frames, frames[1:], strict=False) if b.muted and not a.muted]
    times = [s.t for s in strokes if s.sounded]
    resume = cfg.mute_resume_ms / 1000.0
    unintended = 0
    for t in fired:
        if song is not None and not _in_song(t, song):
            continue
        i = bisect.bisect_right(times, t)
        if i < len(times) and times[i] - t <= resume:
            unintended += 1
    return MuteStats(len(fired), unintended)


# -- judgement ------------------------------------------------------------------------


def _judge(r: TakeReport, cfg: CoachConfig) -> tuple[tuple[Finding, ...], tuple[str, ...]]:
    found: list[Finding] = []
    good: list[str] = []
    s = r.strokes
    n = s.strokes

    if s.missed >= 2 and s.missed / n >= 0.1:
        found.append(
            Finding(
                "missed",
                2.0 * s.missed / n,
                f"{s.missed} of {n} strokes made no sound because the pick grip was not held",
                "keep the thumb pad pressed on the side of your index finger while you strum",
            )
        )
    elif s.missed == 0:
        good.append("every stroke you meant to play sounded")
    if s.grip_drops >= 2 and s.grip_drops / n >= 0.1:
        found.append(
            Finding(
                "grip_drop",
                1.5 * s.grip_drops / n,
                f"in {s.grip_drops} strokes the pick grip let go partway, so only some "
                "strings sounded",
                "hold the pick grip until the stroke has passed all six strings",
            )
        )
    if s.thin / n >= 0.25:
        found.append(
            Finding(
                "thin",
                s.thin / n,
                f"{s.thin} of {n} strokes only brushed {cfg.thin_strings - 1} strings or fewer",
                "swing through the whole width of the head",
            )
        )
    if s.down >= 3 and s.up >= 3:
        if s.up_strings < 0.6 * s.down_strings:
            found.append(
                Finding(
                    "weak_up",
                    0.6 * (1.0 - s.up_strings / s.down_strings),
                    f"up-strums crossed {s.up_strings:.1f} strings on average, down-strums "
                    f"{s.down_strings:.1f}",
                    "let the up-strum travel back across all the strings",
                )
            )
        slow, fast = sorted((s.up_speed, s.down_speed))
        if fast > 0 and slow / fast < 0.6:
            which = "up" if s.up_speed < s.down_speed else "down"
            found.append(
                Finding(
                    "uneven_speed",
                    0.5 * (1.0 - slow / fast),
                    f"{which}-strums were much slower than the others "
                    f"({slow:.1f} against {fast:.1f} head lengths a second)",
                    "make the up and down strokes the same size and speed",
                )
            )
        else:
            good.append("your up and down strokes were even")
    if s.double_hits_blocked >= 3:
        found.append(
            Finding(
                "double_hits",
                min(1.0, s.double_hits_blocked / n),
                f"{s.double_hits_blocked} times a fingertip wobbled back over a string it had "
                "just played (the double-hit guard muted those)",
                "make each stroke one clean pass and pause past the strings",
            )
        )

    if (tm := r.timing) is not None and tm.strokes >= cfg.min_strokes:
        if abs(tm.mean_ms) > 25.0 and tm.on_time_share < 0.75:
            late = tm.mean_ms > 0
            found.append(
                Finding(
                    "drag" if late else "rush",
                    min(1.5, abs(tm.mean_ms) / 80.0) + 0.5 * (1.0 - tm.on_time_share),
                    f"strums landed about {abs(tm.mean_ms):.0f} ms "
                    f"{'behind' if late else 'ahead of'} the beat on average "
                    f"({'dragging' if late else 'rushing'}); {tm.on_time_share:.0%} were within "
                    f"{cfg.on_time_ms:.0f} ms",
                    "count along and land each down-strum on the beat"
                    if late
                    else "relax and let the beat come to you instead of jumping ahead",
                )
            )
        elif tm.on_time_share >= 0.75:
            good.append(f"timing was tight: {tm.on_time_share:.0%} of strums on the beat")
        drift = tm.second_half_ms - tm.first_half_ms
        if abs(drift) > 40.0:
            found.append(
                Finding(
                    "drift",
                    min(1.0, abs(drift) / 120.0),
                    f"you {'fell behind' if drift > 0 else 'pulled ahead'} as the song went on: "
                    f"{tm.first_half_ms:+.0f} ms in the first half, "
                    f"{tm.second_half_ms:+.0f} ms in the second",
                    "keep the same pulse to the end",
                )
            )
    if (tp := r.tempo) is not None:
        change = (tp.end_per_min - tp.start_per_min) / tp.start_per_min
        if abs(change) > 0.12:
            found.append(
                Finding(
                    "tempo_drift",
                    min(1.0, abs(change) * 2.0),
                    f"you {'sped up' if change > 0 else 'slowed down'} from about "
                    f"{tp.start_per_min:.0f} to {tp.end_per_min:.0f} strokes a minute",
                    "pick a steady pace and hold it, tapping your foot helps",
                )
            )
        if tp.unevenness > 0.25:
            found.append(
                Finding(
                    "uneven_rhythm",
                    min(1.0, tp.unevenness * 1.5),
                    f"the gaps between strokes varied by about {tp.unevenness:.0%}",
                    "aim for an even down-up rhythm",
                )
            )

    for c in r.changes:
        bad = c.late + c.missed
        if bad:
            lag = f", about {c.mean_lag_ms:.0f} ms after it was due" if c.mean_lag_ms else ""
            missed = f" and never got there {c.missed} times" if c.missed else ""
            found.append(
                Finding(
                    f"change:{c.frm}>{c.to}",
                    0.8 * bad / c.count + 0.3 * min(1.0, (c.mean_lag_ms or 0.0) / 500.0),
                    f"the change from {c.frm} to {c.to} was late {c.late} of {c.count} "
                    f"times{lag}{missed}"
                    + (
                        f"; {c.wrong_strums} strums still hit the old chord"
                        if c.wrong_strums
                        else ""
                    ),
                    f"start moving to the {c.to} section on the last strum of {c.frm}",
                )
            )
        if c.left_early and c.left_early / c.count >= 0.5 and c.mean_left_early_ms:
            found.append(
                Finding(
                    f"rush_change:{c.frm}>{c.to}",
                    0.5 * c.left_early / c.count,
                    f"you let go of {c.frm} about {c.mean_left_early_ms:.0f} ms before {c.to} "
                    f"was due, cutting {c.frm} short",
                    f"hold {c.frm} through its last beat, then move",
                )
            )
    if r.changes and not any(c.late or c.missed for c in r.changes):
        good.append("every chord change arrived on time")
    if r.chord_accuracy is not None and r.chord_accuracy < 0.8:
        found.append(
            Finding(
                "accuracy",
                1.0 - r.chord_accuracy,
                f"only {r.chord_accuracy:.0%} of the strums were on the chord the song wanted",
                "watch the next-chord cue and move a beat ahead",
            )
        )

    if r.mutes.likely_unintended >= 2:
        found.append(
            Finding(
                "mute",
                0.3 * min(3, r.mutes.likely_unintended),
                f"{r.mutes.likely_unintended} palm mutes fired while you were still playing",
                "keep the strumming hand curled between strokes, rest it flat only to stop",
            )
        )
    if r.racket_lost_share > 0.1:
        found.append(
            Finding(
                "lost",
                r.racket_lost_share,
                f"the racket was lost for {r.racket_lost_share:.0%} of the take",
                "keep it in the frame and press l to lock on again",
            )
        )
    found.sort(key=lambda f: f.score, reverse=True)
    return tuple(found), tuple(good)


def facts(r: TakeReport, top: int = 4) -> dict:
    """A compact, JSON-able summary of the take for the LLM to put into words."""
    s = r.strokes
    out: dict = {
        "seconds": round(r.seconds, 1),
        "song": None
        if r.song is None
        else {"title": r.song.title, "bpm": round(r.song.bpm), "finished": r.song.finished},
        "enough_to_judge": r.enough,
        "strokes": {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(s).items()},
        "chord_accuracy": None if r.chord_accuracy is None else round(r.chord_accuracy, 2),
        "palm_mutes": asdict(r.mutes),
        "racket_lost_share": round(r.racket_lost_share, 2),
        "strengths": list(r.strengths),
        "issues_ranked": [{"fact": f.fact, "tip": f.tip} for f in r.findings[:top]],
    }
    if r.timing is not None:
        out["timing_ms"] = {k: round(v, 2) for k, v in asdict(r.timing).items()}
    if r.tempo is not None:
        out["tempo"] = {k: round(v, 2) for k, v in asdict(r.tempo).items()}
    if r.changes:
        out["chord_changes_worst_first"] = [
            {
                "from": c.frm,
                "to": c.to,
                "times": c.count,
                "late": c.late,
                "missed": c.missed,
                "left_early": c.left_early,
                "mean_lag_ms": None if c.mean_lag_ms is None else round(c.mean_lag_ms),
                "wrong_chord_strums": c.wrong_strums,
            }
            for c in r.changes[:6]
        ]
    return out
