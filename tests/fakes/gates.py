"""Stand-in strum gates and a manual clock, for testing code that consumes them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from racketguitar.types import GripState, HandObs, Onset


@dataclass
class ScriptedGate:
    """A ``StrumGate`` that is open exactly inside the given ``[start, end)`` intervals.

    Records every call so tests can check the consumer used the interface correctly.
    ``ScriptedGate.always()`` and ``ScriptedGate.never()`` cover the common cases.
    """

    open_intervals: Sequence[tuple[float, float]] = ()
    hands: list[tuple[float, HandObs | None]] = field(default_factory=list)
    onsets: list[Onset] = field(default_factory=list)
    emitted: list[float] = field(default_factory=list)
    queries: list[tuple[float, bool]] = field(default_factory=list)

    @classmethod
    def always(cls) -> ScriptedGate:
        return cls([(float("-inf"), float("inf"))])

    @classmethod
    def never(cls) -> ScriptedGate:
        return cls([])

    def observe_hand(self, t: float, hand: HandObs | None) -> None:
        self.hands.append((t, hand))

    def observe_onset(self, onset: Onset) -> None:
        self.onsets.append(onset)

    def note_emitted(self, t: float) -> None:
        self.emitted.append(t)

    def allows(self, t: float) -> bool:
        ok = any(a <= t < b for a, b in self.open_intervals)
        self.queries.append((t, ok))
        return ok

    @property
    def grip(self) -> GripState:
        t = self.hands[-1][0] if self.hands else None
        if t is None:
            return "unknown"
        return "closed" if any(a <= t < b for a, b in self.open_intervals) else "open"


class FakeClock:
    """A monotonic clock that only moves when told to. Call it like ``time.monotonic``."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now
