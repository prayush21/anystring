"""Acceptance report for a recorded session: the PID's on-device checks, measured.

Record each check with ``racketguitar play --record DIR``, then run this on the take.
The session is replayed through a fresh ``Pipeline`` (the same code live play ran), so
every detected crossing is seen with its direction and whether the gate let it play,
not only the notes that sounded. Hits are grouped into strums: one sweep in one
direction, strings in order, no long pause.

    uv run python -m tools.acceptance recordings/idle-1 --check idle
    uv run python -m tools.acceptance recordings/strums-1 --check strums --expected 100
    uv run python -m tools.acceptance recordings/burst-1 --check burst
    uv run python -m tools.acceptance recordings/move-1 --check move
    uv run python -m tools.acceptance recordings/song-1 --check song --json song-1.json

Without ``--check`` it prints the measurements only. Strum-to-sound latency is measured
from a second-device recording by ``tools.latency_probe``; this report shows the app's
internal detect-to-audio log and frame lag alongside, which cover only part of that path.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from racketguitar.pipeline import Pipeline
from racketguitar.replay.harness import compare_events
from racketguitar.replay.record import Session, load_session
from racketguitar.types import Direction

__all__ = [
    "Hit",
    "Strum",
    "Take",
    "Verdict",
    "CHECKS",
    "measure",
    "group_strums",
    "verdicts",
    "report",
    "main",
]

# Strum grouping: a pause this long between hits, a change of direction, or a string
# out of sweep order starts a new strum.
STRUM_GAP_S = 0.10
DROP_FACTOR = 1.5  # in a burst, an interval this many times the median is a drop

# PID targets (Objectives and Testing and acceptance).
IDLE_MIN_S = 120.0
IDLE_MAX_FALSE = 2  # "2 minutes holding the racket without strumming: under 2 false notes"
DETECTED_MIN = 0.95
UP_MIN = 0.90
BURST_RATE = 8.0
BURST_WINDOW_S = 10.0
TRACKED_MIN = 0.98

Status = Literal["PASS", "FAIL", "CHECK"]
CHECKS = ("idle", "strums", "burst", "move", "song")


@dataclass(frozen=True)
class Hit:
    t: float
    string: int
    direction: Direction
    played: bool
    chord: str | None  # active chord when it was hit, None = open strings


@dataclass(frozen=True)
class Strum:
    t: float  # first hit
    t_end: float  # last hit
    direction: Direction
    strings: tuple[int, ...]  # in hit order
    notes: int  # hits that sounded
    voiced: int  # strings the chord plays
    chord: str | None

    @property
    def played(self) -> bool:
        return self.notes > 0

    @property
    def full(self) -> bool:
        """Every string the chord plays sounded."""
        return self.notes >= self.voiced


@dataclass(frozen=True)
class Take:
    path: str
    pose_source: str
    duration_s: float
    frames: int
    fps: float
    dropped_frames: int  # missing frames inferred from gaps in capture times
    lag_ms: tuple[float, float]  # capture -> processing start, p50 / p95
    detect_to_audio_ms: dict | None  # the app's own log, from session.json
    locked_after_s: float | None  # first tracked frame, from the take's start
    tracked: float  # fraction of frames since the first lock with the racket tracked
    strumming_hand: float  # fraction of tracked frames with a strumming hand
    armed: float  # fraction of strumming-hand frames classified closed (pick grip)
    grip_flips_per_min: float
    origin_travel_hl: float  # extent of the throat's motion, head lengths
    angle_range_deg: float
    notes: int  # note-ons played
    damps: int
    replay_matches: bool
    mismatches: list[str]
    chord_names: tuple[str, ...]
    strums: list[Strum] = field(repr=False)

    @property
    def played(self) -> list[Strum]:
        return [s for s in self.strums if s.played]

    @property
    def blocked(self) -> list[Strum]:
        """Crossings the gate (or an unplayed chord) kept silent."""
        return [s for s in self.strums if not s.played]


@dataclass(frozen=True)
class Verdict:
    check: str
    measured: str
    target: str
    status: Status


def measure(session: Session | str | Path) -> Take:
    s = load_session(session) if isinstance(session, (str, Path)) else session
    path = str(session) if isinstance(session, (str, Path)) else ""
    pipe = Pipeline(s.config, s.calibration)
    events, hits = [], []
    seen: set[tuple[float, int, str]] = set()
    visible = hand = closed = 0
    first_visible: int | None = None
    grips: list[str] = []
    origins, angles = [], []
    for i, rec in enumerate(s.frames):
        events += pipe.step(rec.inp)
        snap = pipe.snapshot()
        for r in snap.recent:
            key = (r.hit.t, r.hit.string, r.hit.direction)
            if key not in seen:
                seen.add(key)
                hits.append(Hit(r.hit.t, r.hit.string, r.hit.direction, r.played, snap.chord))
        if snap.visible:
            visible += 1
            if first_visible is None:
                first_visible = i
            p = rec.inp.pose
            origins.append((p.origin.x / p.head_len, p.origin.y / p.head_len))
            angles.append(math.atan2(p.u.y, p.u.x))
            if snap.strumming is not None:
                hand += 1
                closed += snap.grip == "closed"
        if snap.grip != "unknown":
            grips.append(snap.grip)
    voiced = _voiced_by_chord(s)
    strums = group_strums(hits, voiced)

    t = np.array([f.inp.t for f in s.frames])
    dt = np.diff(t)
    med = float(np.median(dt)) if dt.size else 0.0
    dropped = int(np.sum(np.maximum(np.round(dt / med) - 1, 0))) if med > 0 else 0
    lag = np.array([(f.inp.proc_t - f.inp.t) * 1000.0 for f in s.frames])
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    flips = sum(a != b for a, b in zip(grips, grips[1:], strict=False))
    o = np.array(origins).reshape(-1, 2)
    a = np.unwrap(np.array(angles))
    mismatches = compare_events(s.events, events)
    return Take(
        path=path,
        pose_source=str(s.meta.get("pose_source", "flow")),
        duration_s=duration,
        frames=len(s.frames),
        fps=1.0 / med if med > 0 else 0.0,
        dropped_frames=dropped,
        lag_ms=(_pct(lag, 50), _pct(lag, 95)),
        detect_to_audio_ms=s.meta.get("detect_to_audio_ms"),
        locked_after_s=None if first_visible is None else float(t[first_visible] - t[0]),
        tracked=visible / (len(s.frames) - first_visible) if first_visible is not None else 0.0,
        strumming_hand=hand / visible if visible else 0.0,
        armed=closed / hand if hand else 0.0,
        grip_flips_per_min=flips * 60.0 / duration if duration > 0 else 0.0,
        origin_travel_hl=float(np.linalg.norm(o.max(0) - o.min(0))) if len(o) else 0.0,
        angle_range_deg=float(np.degrees(a.max() - a.min())) if a.size else 0.0,
        notes=sum(e.kind == "on" for e in events),
        damps=sum(e.kind == "damp" for e in events),
        replay_matches=not mismatches,
        mismatches=mismatches,
        chord_names=tuple(name for name, _ in s.config.chords.chords),
        strums=strums,
    )


def group_strums(hits: list[Hit], voiced: dict[str | None, int]) -> list[Strum]:
    """Split time-ordered hits into sweeps: same direction, strings in sweep order
    (down = rising string index), no pause over ``STRUM_GAP_S``."""
    groups: list[list[Hit]] = []
    for h in sorted(hits, key=lambda h: h.t):
        g = groups[-1] if groups else None
        if (
            g is None
            or h.direction != g[-1].direction
            or h.t - g[-1].t > STRUM_GAP_S
            or (h.string <= g[-1].string if h.direction == "down" else h.string >= g[-1].string)
        ):
            groups.append([h])
        else:
            g.append(h)
    out = []
    for g in groups:
        chord = next((h.chord for h in g if h.played), g[0].chord)
        out.append(
            Strum(
                t=g[0].t,
                t_end=g[-1].t,
                direction=g[0].direction,
                strings=tuple(h.string for h in g),
                notes=sum(h.played for h in g),
                voiced=voiced.get(chord, 6),
                chord=chord,
            )
        )
    return out


def verdicts(
    take: Take,
    check: str,
    *,
    expected: int = 100,
    expected_up: int | None = None,
    rate: float = BURST_RATE,
    window_s: float = BURST_WINDOW_S,
) -> list[Verdict]:
    played = take.played
    if check == "idle":
        n = len(played)
        long_enough = take.duration_s >= IDLE_MIN_S
        status: Status = "FAIL" if n >= IDLE_MAX_FALSE else "PASS" if long_enough else "CHECK"
        return [
            Verdict(
                "false strums while idle",
                f"{n} strums ({take.notes} notes) in {take.duration_s:.0f} s"
                + ("" if long_enough else f", take shorter than {IDLE_MIN_S:.0f} s"),
                f"under {IDLE_MAX_FALSE} in {IDLE_MIN_S:.0f} s",
                status,
            )
        ]
    if check == "strums":
        up_exp = expected // 2 if expected_up is None else expected_up
        down_exp = expected - up_exp
        downs = sum(s.direction == "down" for s in played)
        ups = len(played) - downs
        found = min(downs, down_exp) + min(ups, up_exp)
        extra = max(0, downs - down_exp) + max(0, ups - up_exp)
        breaks = sum(a.direction == b.direction for a, b in zip(played, played[1:], strict=False))
        return [
            Verdict(
                "strums detected",
                f"{found}/{expected} ({found / expected:.0%}); {downs} down, {ups} up, "
                f"{extra} extra",
                f"{DETECTED_MIN:.0%} or more",
                "PASS" if found >= DETECTED_MIN * expected else "FAIL",
            ),
            Verdict(
                "up-strums correct",
                f"{min(ups, up_exp)}/{up_exp} ({min(ups, up_exp) / up_exp:.0%}); "
                f"{breaks} down/up alternation breaks",
                f"{UP_MIN:.0%} or more",
                "PASS" if min(ups, up_exp) >= UP_MIN * up_exp else "FAIL",
            ),
        ]
    if check == "burst":
        need = math.ceil(rate * window_s)
        best, gaps, t0 = _best_window(played, window_s)
        status = "PASS" if best >= need and gaps == 0 else "FAIL"
        return [
            Verdict(
                f"sustained {rate:g}/s for {window_s:g} s",
                f"best window: {best} strums from t+{t0:.1f} s, {gaps} drops "
                f"(intervals over {DROP_FACTOR}x the median)",
                f"{need} strums, no drops",
                status,
            )
        ]
    if check == "move":
        # Optical flow can drift onto the background without ever losing track, so a
        # held track is necessary but not proof: the take needs watching for drift.
        moved = take.origin_travel_hl >= 0.5 or take.angle_range_deg >= 20.0
        note = "check the strings stayed on the head" if moved else "racket barely moved"
        return [
            Verdict(
                "tracking while moving",
                f"{take.tracked:.1%} of frames since the lock; racket moved "
                f"{take.origin_travel_hl:.2f} head lengths, turned "
                f"{take.angle_range_deg:.0f} deg; {note}",
                f"{TRACKED_MIN:.0%} of frames, no drift",
                "CHECK" if take.tracked >= TRACKED_MIN else "FAIL",
            )
        ]
    if check == "song":
        seq = _collapse([s.chord for s in played])
        order = [c for c in take.chord_names]
        in_order = _is_subsequence(order, seq)
        shown = " ".join(c or "open" for c in seq[:24]) + (" ..." if len(seq) > 24 else "")
        return [
            Verdict(
                "four-chord song",
                f"chords played: {shown or 'none'}",
                f"{' '.join(order)} in order; then listen to the replayed WAV",
                "CHECK" if in_order else "FAIL",
            )
        ]
    raise ValueError(f"unknown check: {check}")


def report(take: Take, results: list[Verdict]) -> str:
    played, blocked = take.played, take.blocked
    downs = sum(s.direction == "down" for s in played)
    full = sum(s.full for s in played)
    lines = [
        f"{take.path or 'session'}  ({take.pose_source} poses)",
        f"  {take.duration_s:.1f} s, {take.frames} frames at {take.fps:.1f} fps, "
        f"{take.dropped_frames} dropped",
        f"  frame lag (capture -> processing) p50 {take.lag_ms[0]:.1f} ms, "
        f"p95 {take.lag_ms[1]:.1f} ms",
    ]
    d = take.detect_to_audio_ms
    if d:
        lines.append(
            f"  detect-to-audio (app log) p50 {d['p50']:.1f} ms, p95 {d['p95']:.1f} ms "
            f"over {d['n']} notes"
        )
    lock = (
        "never locked"
        if take.locked_after_s is None
        else (f"locked after {take.locked_after_s:.1f} s, then tracked {take.tracked:.1%}")
    )
    lines += [
        f"  racket {lock}; strumming hand in {take.strumming_hand:.1%} "
        f"of those frames, armed {take.armed:.1%} of the time; "
        f"{take.grip_flips_per_min:.1f} grip flips/min",
        f"  {len(played)} strums played ({downs} down, {len(played) - downs} up, "
        f"{full} with every chord string), {len(blocked)} crossings kept silent; "
        f"{take.notes} notes, {take.damps} damps",
    ]
    if not take.replay_matches:
        lines.append(
            "  WARNING: replay differs from the recording (code or settings changed "
            "since it was made); numbers are from the replay"
        )
    for v in results:
        lines.append(f"  [{v.status}] {v.check}: {v.measured}  (target: {v.target})")
    return "\n".join(lines)


def _best_window(played: list[Strum], window_s: float) -> tuple[int, int, float]:
    """Most strums in any ``window_s`` window: (count, drops inside it, start offset)."""
    if not played:
        return 0, 0, 0.0
    t = np.array([s.t for s in played])
    best_i, best_n = 0, 0
    for i in range(t.size):
        n = int(np.searchsorted(t, t[i] + window_s, side="left")) - i
        if n > best_n:
            best_i, best_n = i, n
    iv = np.diff(t[best_i : best_i + best_n])
    gaps = int(np.sum(iv > DROP_FACTOR * np.median(iv))) if iv.size else 0
    return best_n, gaps, float(t[best_i] - t[0])


def _voiced_by_chord(s: Session) -> dict[str | None, int]:
    out: dict[str | None, int] = {None: s.config.strings.count}
    for name, notes in s.config.chords.chords:
        out[name] = sum(n is not None for n in notes)
    return out


def _collapse(xs: list) -> list:
    return [x for i, x in enumerate(xs) if i == 0 or x != xs[i - 1]]


def _is_subsequence(want: list, seq: list) -> bool:
    it = iter(seq)
    return all(any(x == w for x in it) for w in want)


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else 0.0


def _to_json(take: Take, results: list[Verdict]) -> dict:
    d = asdict(take)
    d["strums"] = [{**asdict(s), "played": s.played, "full": s.full} for s in take.strums]
    d["verdicts"] = [asdict(v) for v in results]
    return d


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="acceptance", description="Acceptance report for a take.")
    p.add_argument("session", help="recorded session directory")
    p.add_argument("--check", choices=CHECKS, help="judge the take against this check")
    p.add_argument("--expected", type=int, default=100, help="strums performed (strums)")
    p.add_argument("--expected-up", type=int, help="up-strums performed (default half)")
    p.add_argument("--rate", type=float, default=BURST_RATE, help="strums per second (burst)")
    p.add_argument("--window", type=float, default=BURST_WINDOW_S, help="seconds (burst)")
    p.add_argument("--json", help="also write the full report, every strum included, here")
    args = p.parse_args(argv)
    take = measure(args.session)
    results = (
        verdicts(
            take,
            args.check,
            expected=args.expected,
            expected_up=args.expected_up,
            rate=args.rate,
            window_s=args.window,
        )
        if args.check
        else []
    )
    print(report(take, results))
    if args.json:
        Path(args.json).write_text(json.dumps(_to_json(take, results), indent=2) + "\n")
    return 1 if any(v.status == "FAIL" for v in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
