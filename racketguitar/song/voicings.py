"""Chord names -> one MIDI note per string, for the handle sections.

A name is a root (``A``..``G`` with ``#``/``b``), a quality (``m``, ``7``, ``maj7``,
``sus4``, ...) and an optional slash bass (``D/F#``). Common chords use a table of
open-position shapes; anything else uses a movable barre shape (E or A form) at the
lowest fret that fits. A slash bass goes on the low E or A string when it is within
reach of the shape. Every voicing sounds only chord tones (plus the slash bass), which
``tests/song`` checks over every quality.

Shapes are fret strings, low E first, ``x`` = string not played. A capo adds its fret
to every note, so names stay the shapes a chord sheet shows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Chord",
    "parse_chord",
    "is_chord",
    "chord_tones",
    "voicing",
    "shape_to_midi",
    "UnknownChordError",
]

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Quality -> semitones above the root.
INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "m": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "m7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "7sus4": (0, 5, 7, 10),
    "dim": (0, 3, 6),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "aug": (0, 4, 8),
    "5": (0, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10, 2),
    "add9": (0, 4, 7, 2),
}

# Spellings -> quality. Extensions we have no shape for fall back to their core chord.
ALIASES: dict[str, str] = {
    "": "maj",
    "maj": "maj",
    "M": "maj",
    "m": "m",
    "min": "m",
    "-": "m",
    "7": "7",
    "dom7": "7",
    "m7": "m7",
    "min7": "m7",
    "-7": "m7",
    "maj7": "maj7",
    "M7": "maj7",
    "Δ": "maj7",
    "Δ7": "maj7",
    "sus": "sus4",
    "sus4": "sus4",
    "sus2": "sus2",
    "2": "add9",
    "add2": "add9",
    "add9": "add9",
    "7sus": "7sus4",
    "7sus4": "7sus4",
    "dim": "dim",
    "°": "dim",
    "o": "dim",
    "dim7": "dim7",
    "°7": "dim7",
    "o7": "dim7",
    "m7b5": "m7b5",
    "ø": "m7b5",
    "ø7": "m7b5",
    "aug": "aug",
    "+": "aug",
    "5": "5",
    "6": "6",
    "m6": "m6",
    "9": "9",
    "maj9": "maj7",
    "m9": "m7",
    "madd9": "m",
    "11": "7",
    "13": "7",
    "m11": "m7",
    "7b9": "7",
    "7#9": "7",
}

# Open-position shapes for common chords, keyed by (root pitch class, quality).
OPEN: dict[tuple[int, str], str] = {
    (0, "maj"): "x32010",
    (0, "7"): "x32310",
    (0, "maj7"): "x32000",
    (0, "add9"): "x32030",
    (2, "maj"): "xx0232",
    (2, "m"): "xx0231",
    (2, "7"): "xx0212",
    (2, "maj7"): "xx0222",
    (2, "m7"): "xx0211",
    (2, "sus2"): "xx0230",
    (2, "sus4"): "xx0233",
    (4, "maj"): "022100",
    (4, "m"): "022000",
    (4, "7"): "020100",
    (4, "m7"): "022030",
    (4, "maj7"): "021100",
    (4, "sus4"): "022200",
    (5, "maj"): "133211",
    (5, "maj7"): "xx3210",
    (7, "maj"): "320003",
    (7, "7"): "320001",
    (7, "maj7"): "320002",
    (9, "maj"): "x02220",
    (9, "m"): "x02210",
    (9, "7"): "x02020",
    (9, "m7"): "x02010",
    (9, "maj7"): "x02120",
    (9, "sus2"): "x02200",
    (9, "sus4"): "x02230",
    (9, "7sus4"): "x02030",
    (11, "7"): "x21202",
}

# Movable shapes: fret offsets from the root's fret, None = string not played.
# E form has the root on the low E string, A form on the A string.
E_FORM: dict[str, tuple[int | None, ...]] = {
    "maj": (0, 2, 2, 1, 0, 0),
    "m": (0, 2, 2, 0, 0, 0),
    "7": (0, 2, 0, 1, 0, 0),
    "m7": (0, 2, 0, 0, 0, 0),
    "maj7": (0, None, 1, 1, 0, None),
    "sus4": (0, 2, 2, 2, 0, 0),
    "7sus4": (0, 2, 0, 2, 0, 0),
    "dim": (0, 1, 2, 0, None, None),
    "aug": (0, None, 2, 1, 1, 0),
    "5": (0, 2, 2, None, None, None),
    "6": (0, 2, 2, 1, 2, 0),
    "m6": (0, 2, 2, 0, 2, 0),
    "9": (0, 2, 0, 1, 0, 2),
}
A_FORM: dict[str, tuple[int | None, ...]] = {
    "maj": (None, 0, 2, 2, 2, 0),
    "m": (None, 0, 2, 2, 1, 0),
    "7": (None, 0, 2, 0, 2, 0),
    "m7": (None, 0, 2, 0, 1, 0),
    "maj7": (None, 0, 2, 1, 2, 0),
    "sus2": (None, 0, 2, 2, 0, 0),
    "sus4": (None, 0, 2, 2, 3, 0),
    "7sus4": (None, 0, 2, 0, 3, 0),
    "dim": (None, 0, 1, 2, 1, None),
    "dim7": (None, 0, 1, 2, 1, 2),
    "m7b5": (None, 0, 1, 0, 1, None),
    "aug": (None, 0, 3, 2, 2, 1),
    "5": (None, 0, 2, 2, None, None),
    "6": (None, 0, 2, 2, 2, 2),
    "m6": (None, 0, 2, 2, 1, 2),
    "9": (None, 0, -1, 0, 0, 0),
    "add9": (None, 0, 2, 4, 2, 0),
}
STANDARD_PCS = (4, 9, 2, 7, 11, 4)  # open strings E A D G B E

_NAME = re.compile(r"^([A-G])([#b]?)([^/]*)(?:/([A-G])([#b]?))?$")


class UnknownChordError(ValueError):
    pass


@dataclass(frozen=True)
class Chord:
    name: str  # as written
    root: int  # pitch class
    quality: str  # a key of INTERVALS
    bass: int | None  # slash bass pitch class, None = root

    @property
    def base_name(self) -> str:
        """The name without its slash bass (``C/B`` -> ``C``)."""
        return self.name.split("/")[0]


def _pc(letter: str, accidental: str) -> int:
    return (NOTE_PC[letter] + {"#": 1, "b": -1, "": 0}[accidental]) % 12


def parse_chord(name: str) -> Chord:
    m = _NAME.match(name.strip())
    if not m:
        raise UnknownChordError(f"not a chord: {name!r}")
    letter, acc, qual, bass_letter, bass_acc = m.groups()
    if qual not in ALIASES:
        raise UnknownChordError(f"unknown chord quality {qual!r} in {name!r}")
    bass = _pc(bass_letter, bass_acc) if bass_letter else None
    return Chord(name.strip(), _pc(letter, acc), ALIASES[qual], bass)


def is_chord(token: str) -> bool:
    try:
        parse_chord(token)
    except UnknownChordError:
        return False
    return True


def chord_tones(chord: Chord) -> frozenset[int]:
    """Pitch classes the chord may sound (its tones plus a slash bass)."""
    tones = {(chord.root + i) % 12 for i in INTERVALS[chord.quality]}
    if chord.bass is not None:
        tones.add(chord.bass)
    return frozenset(tones)


def shape_to_midi(
    shape: str | tuple[int | None, ...], tuning: tuple[int, ...], capo: int = 0
) -> tuple[int | None, ...]:
    frets = _frets(shape) if isinstance(shape, str) else shape
    if len(frets) != len(tuning):
        raise ValueError(f"shape {shape!r} needs {len(tuning)} strings")
    return tuple(None if f is None else n + f + capo for n, f in zip(tuning, frets, strict=True))


def voicing(name: str | Chord, tuning: tuple[int, ...], capo: int = 0) -> tuple[int | None, ...]:
    """MIDI note per string (low E first) for a chord name; None = string not played."""
    chord = parse_chord(name) if isinstance(name, str) else name
    if len(tuning) != 6 or tuple(n % 12 for n in tuning) != STANDARD_PCS:
        raise ValueError("chord shapes need six strings in standard tuning")
    frets = _shape(chord)
    if chord.bass is not None and chord.bass != chord.root:
        frets = _with_bass(frets, chord.bass, tuning)
    return shape_to_midi(frets, tuning, capo)


def _frets(shape: str) -> tuple[int | None, ...]:
    return tuple(None if c in "xX" else int(c) for c in shape)


def _shape(chord: Chord) -> tuple[int | None, ...]:
    if (chord.root, chord.quality) in OPEN:
        return _frets(OPEN[(chord.root, chord.quality)])
    best: tuple[int | None, ...] | None = None
    for form, string in ((E_FORM, 0), (A_FORM, 1)):
        offsets = form.get(chord.quality)
        if offsets is None:
            continue
        r = (chord.root - STANDARD_PCS[string]) % 12
        for fret in (r, r + 12):
            frets = tuple(None if o is None else fret + o for o in offsets)
            if all(f is None or f >= 0 for f in frets) and (
                best is None or max(f for f in frets if f is not None) < _top(best)
            ):
                best = frets
                break
    if best is None:  # every quality has at least one form
        raise UnknownChordError(f"no shape for {chord.name!r}")
    return best


def _top(frets: tuple[int | None, ...]) -> int:
    return max(f for f in frets if f is not None)


def _with_bass(
    frets: tuple[int | None, ...], bass: int, tuning: tuple[int, ...]
) -> tuple[int | None, ...]:
    """Put the slash bass on the low E or A string if a finger can reach it."""
    fretted = [f for f in frets if f]
    lo, hi = (min(fretted), max(fretted)) if fretted else (0, 0)
    for string in (0, 1):
        for f in range(0, 13):
            if (tuning[string] + f) % 12 != bass:
                continue
            if f == 0 or lo - 1 <= f <= hi + 1:
                out = list(frets)
                out[string] = f
                if string == 1:
                    out[0] = None
                return tuple(out)
    return frets
