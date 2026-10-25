"""Try pick-grip gate thresholds on recorded takes and rank them.

Record labelled takes, then sweep::

    uv run anystring --record recordings/grip-closed    # pick grip both ways
    uv run anystring --record recordings/grip-relaxed   # pick-grip down, relaxed return
    uv run python -m tools.grip_sweep --closed recordings/grip-closed \\
        --relaxed recordings/grip-relaxed

A take's ``-part2``, ``-part3``, ... folders (the app splits a recording when settings
change, such as after locking on) are included with it; parts with no string crossings
are skipped.

* ``--closed``: every string crossing in the take should play, down and up.
* ``--relaxed``: down-strokes should play; up-strokes are the relaxed return and should
  stay silent.

Each take is replayed once with the gate open, to collect every crossing of a voiced
string and the strumming hand's grip score on every frame. Strum, chord and mute logic
never look at the gate, so those crossings are the same under any setting; each setting
then reruns only the grip classifier, which decides whether a crossing plays. Thresholds
are shares of the take's calibrated pick-grip -> open-hand span (today's defaults when
its calibration is missing or from an older grip score), as in ``[gate]
grip_arm_below`` / ``grip_release_above``. The mic gate is not modelled.

Per setting, ``missed`` counts crossings that should have played but were blocked and
``extra`` counts relaxed returns that played. Settings are ranked by missed + extra
(each is one wrong-sounding moment), then by extra.
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from anystring.config import GateConfig
from anystring.pipeline import Pipeline
from anystring.play.gate import GripClassifier, grip_score, grip_thresholds
from anystring.replay.record import load_session

__all__ = ["Take", "Setting", "Result", "take_parts", "load_take", "evaluate", "sweep", "main"]

Label = Literal["closed", "relaxed"]

ARM = (-0.5, -0.25, 0.0, 0.1, 0.25)
OFF = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75)
HOLD = (1, 2, 3)
RELEASE = (1, 2)


@dataclass(frozen=True, eq=False)
class Take:
    name: str
    label: Label
    scores: np.ndarray  # strumming hand's grip score per frame, NaN = no strumming hand
    hits: tuple[tuple[int, str], ...]  # (frame index, "down"/"up") per voiced crossing
    span: tuple[float, float]  # (closed, open) scores the thresholds are shares of

    def share(self, score: float) -> float:
        c, o = self.span
        return (score - c) / (o - c)


@dataclass(frozen=True)
class Setting:
    arm: float  # grip_arm_below
    off: float  # grip_release_above
    hold: int  # grip_hold_frames
    release: int  # grip_release_frames

    def label(self) -> str:
        return f"arm<{self.arm:+.0%} off>{self.off:.0%} hold {self.hold} release {self.release}"


@dataclass(frozen=True)
class Result:
    setting: Setting
    missed: int  # should have played, blocked
    should_play: int
    extra: int  # relaxed returns that played
    should_block: int

    @property
    def errors(self) -> int:
        return self.missed + self.extra


def take_parts(path: str | Path) -> list[Path]:
    """``path`` and its ``-part2``, ``-part3``, ... siblings that exist, in order."""
    base = Path(path)
    parts = [base] if base.exists() else []
    n = 2
    while (part := base.parent / f"{base.name}-part{n}").exists():
        parts.append(part)
        n += 1
    return parts


def load_take(path: str | Path, label: Label) -> Take:
    """Replay a recorded session with the gate open; keep crossings and grip scores."""
    s = load_session(path)
    cfg = s.config.with_overrides({"gate": {"pick_grip": False, "mic": False}})
    pipe = Pipeline(cfg, s.calibration)
    seen, kept = set(), []  # kept holds every record, so no id is reused
    hits: list[tuple[int, str]] = []
    scores = np.full(len(s.frames), math.nan)
    for i, frame in enumerate(s.frames):
        pipe.step(frame.inp)
        snap = pipe.snapshot()
        for rec in snap.recent:
            if id(rec) not in seen:
                seen.add(id(rec))
                kept.append(rec)
                if rec.midi is not None:
                    hits.append((i, rec.hit.direction))
        if snap.strumming is not None:
            scores[i] = grip_score(snap.strumming)
    # Today's defaults, not the session's: a take recorded before a grip-score change saved
    # defaults on the old score's scale (its calibration is ignored for the same reason).
    span = grip_thresholds(s.calibration, GateConfig(grip_arm_below=0.0, grip_release_above=1.0))
    return Take(Path(path).name, label, scores, tuple(hits), span)


def evaluate(takes: Sequence[Take], setting: Setting) -> Result:
    missed = should_play = extra = should_block = 0
    for take in takes:
        clf = GripClassifier(setting.hold, release_frames=setting.release)
        c, o = take.span
        clf.close_below = c + setting.arm * (o - c)
        clf.open_above = c + setting.off * (o - c)
        armed = [clf.update_score(float(x)) == "closed" for x in take.scores]
        for i, direction in take.hits:
            should = take.label == "closed" or direction == "down"
            if should:
                should_play += 1
                missed += not armed[i]
            else:
                should_block += 1
                extra += armed[i]
    return Result(setting, missed, should_play, extra, should_block)


def sweep(
    takes: Sequence[Take],
    arms: Iterable[float] = ARM,
    offs: Iterable[float] = OFF,
    holds: Iterable[int] = HOLD,
    releases: Iterable[int] = RELEASE,
) -> list[Result]:
    """Every setting with arm below off, best first."""
    results = [
        evaluate(takes, Setting(a, o, h, r))
        for a, o, h, r in itertools.product(arms, offs, holds, releases)
        if a < o
    ]
    return sorted(results, key=lambda r: (r.errors, r.extra, r.missed))


def _row(r: Result, note: str = "") -> str:
    def pct(n: int, d: int) -> str:
        return f"{n:4d}/{d:<4d} ({n / d:4.0%})" if d else f"{n:4d}/0".ljust(16)

    return (
        f"{r.setting.label():38s} missed {pct(r.missed, r.should_play)}"
        f"  extra {pct(r.extra, r.should_block)}  errors {r.errors:4d}  {note}"
    )


def _spread(takes: Sequence[Take]) -> list[str]:
    """Grip score (share of span) at crossing frames, per take and direction."""
    lines = []
    for take in takes:
        for direction in ("down", "up"):
            v = np.array([take.share(take.scores[i]) for i, d in take.hits if d == direction])
            v = v[np.isfinite(v)]
            if v.size:
                qs = (5, 25, 50, 75, 95)
                p = np.percentile(v, qs)
                lines.append(
                    f"  {take.name:22s} {take.label:7s} {direction:4s} n={v.size:4d}  "
                    + "  ".join(f"p{q} {x:+.0%}" for q, x in zip(qs, p, strict=True))
                )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--closed", nargs="+", default=[], help="takes where every crossing should play"
    )
    ap.add_argument(
        "--relaxed", nargs="+", default=[], help="takes of pick-grip downs and relaxed returns"
    )
    ap.add_argument("--top", type=int, default=15, help="settings to list")
    ap.add_argument("--arm", type=float, nargs="+", default=ARM, help="grip_arm_below values")
    ap.add_argument("--off", type=float, nargs="+", default=OFF, help="grip_release_above values")
    ap.add_argument("--hold", type=int, nargs="+", default=HOLD, help="grip_hold_frames values")
    ap.add_argument(
        "--release", type=int, nargs="+", default=RELEASE, help="grip_release_frames values"
    )
    args = ap.parse_args(argv)
    if not args.closed and not args.relaxed:
        ap.error("give at least one --closed or --relaxed take")
    takes: list[Take] = []
    for label, paths in (("closed", args.closed), ("relaxed", args.relaxed)):
        found = 0
        for path in paths:
            parts = take_parts(path)
            if not parts:
                ap.error(f"no recording at {path}")
            for part in parts:
                take = load_take(part, label)
                print(f"{label:7s} {part}: {len(take.hits)} string crossings")
                if take.hits:
                    takes.append(take)
                    found += len(take.hits)
        if paths and not found:
            ap.error(f"the --{label} takes have no string crossings to score")
    print()

    print("grip score at each crossing, as a share of the calibrated pick-grip -> open span:")
    print("\n".join(_spread(takes)))
    results = sweep(takes, args.arm, args.off, args.hold, args.release)
    print(f"\nbest {min(args.top, len(results))} of {len(results)} settings:")
    for r in results[: args.top]:
        print(_row(r))
    g = GateConfig()
    default = Setting(
        g.grip_arm_below, g.grip_release_above, g.grip_hold_frames, g.grip_release_frames
    )
    for name, s in (("(default)", default), ("(old thresholds)", Setting(0.25, 0.75, 2, 2))):
        print(_row(evaluate(takes, s), name))
    if not args.relaxed:
        print("\nno --relaxed take: 'extra' cannot be measured, so lower 'off' never looks worse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
