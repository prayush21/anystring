"""Chord selection from which handle section the fretting hand grips.

The handle (t = 0 at the head end, 1 at the butt) is split into one equal section per
chord in ``ChordConfig.chords``, head -> butt (C, G, Am, F by default). The fretting
hand's knuckle position picks the section, with ``ChordConfig.hysteresis`` (t units)
around the current section's edges. Finger shapes are never read. No fretting hand
means open strings (``StringsConfig.tuning``), after ``ChordConfig.fret_loss_hold_ms``
so a one-frame tracking dropout does not flip a strum to open strings.
"""

from __future__ import annotations

from racketguitar.config import ChordConfig, StringsConfig

__all__ = ["ChordSelector"]


class ChordSelector:
    """Tracks the active handle section and returns the voicing to play."""

    def __init__(self, chords: ChordConfig, strings: StringsConfig) -> None:
        self._count = strings.count
        self._names: tuple[str, ...] = ()
        self._voicings: tuple[tuple[int | None, ...], ...] = ()
        self.set_chords(chords.chords)
        self._open = tuple(strings.tuning)
        self._h = chords.hysteresis
        self._hold = chords.fret_loss_hold_ms / 1000.0
        self._zone: int | None = None
        self._last_seen = -float("inf")

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def voicings_all(self) -> tuple[tuple[int | None, ...], ...]:
        """Voicing of every section, matching ``names``."""
        return self._voicings

    @property
    def zone(self) -> int | None:
        """Active section index (0 = nearest the head), or None for open strings."""
        return self._zone

    @property
    def chord(self) -> str | None:
        """Active chord name, or None for open strings."""
        return None if self._zone is None else self._names[self._zone]

    @property
    def voicing(self) -> tuple[int | None, ...]:
        """MIDI note per string (low E first); None = string not played."""
        return self._open if self._zone is None else self._voicings[self._zone]

    def set_chords(self, chords: tuple[tuple[str, tuple[int | None, ...]], ...]) -> None:
        """Relabel the sections (song mode) without touching the hand state. With the
        same section count the active section stays; otherwise it is found afresh."""
        if not chords:
            raise ValueError("chords.chords must not be empty")
        for name, notes in chords:
            if len(notes) != self._count:
                raise ValueError(f"chord {name!r} needs {self._count} notes")
        if len(chords) != len(self._names):
            self._zone = None
        self._names = tuple(name for name, _ in chords)
        self._voicings = tuple(tuple(notes) for _, notes in chords)

    def reset(self) -> None:
        self._zone = None
        self._last_seen = -float("inf")

    def update(self, t: float, handle_t: float | None) -> int | None:
        """Feed the fretting knuckle's handle position (None = no fretting hand)."""
        if handle_t is None:
            if self._zone is not None and t - self._last_seen > self._hold:
                self._zone = None
            return self._zone
        self._last_seen = t
        n = len(self._names)
        x = min(1.0, max(0.0, handle_t))
        z = self._zone
        if z is None or not (z / n - self._h <= x <= (z + 1) / n + self._h):
            self._zone = min(int(x * n), n - 1)
        return self._zone
