"""Chord sheets -> a timed song: lyric lines, chords and when each starts.

Two layouts are read, and may be mixed in one file:

* ChordPro: chords in brackets inside the lyric, ``[Am]Folding paper [D]boats``.
* Chords above lyrics (what chord sites show): a line of only chord names, then the
  lyric line it sits over. Column positions place the chords on the words.

Directives ``{title: ...}``, ``{artist: ...}``, ``{tempo: 169}``, ``{time: 4/4}``,
``{capo: 2}`` and ``{chord_bars: 0.5}`` (default chord length) can appear anywhere.
Sections are ``[Verse 1]`` lines (or ``{start_of_chorus}`` ...). A chord lasts
``chord_bars`` bars (else ``SongConfig.chord_bars``) unless written ``Am*2`` (2 bars).
Without a sheet tempo, 100 BPM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from racketguitar.song.voicings import UnknownChordError, parse_chord

__all__ = ["ChordSpan", "Line", "Song", "parse_sheet", "load_sheet", "SHEET_SUFFIXES"]

SHEET_SUFFIXES = (".txt", ".cho", ".chopro", ".chordpro", ".crd")
DEFAULT_TEMPO = 100.0

_DIRECTIVE = re.compile(r"^\{\s*([A-Za-z_]+)\s*(?::\s*(.*?))?\s*\}$")
_INLINE = re.compile(r"\[([^\]]+)\]")
_LENGTH = re.compile(r"^(.+?)\*(\d+(?:\.\d+)?)$")
_SECTION_DIRECTIVES = {
    "start_of_verse": "Verse",
    "sov": "Verse",
    "start_of_chorus": "Chorus",
    "soc": "Chorus",
    "start_of_bridge": "Bridge",
    "sob": "Bridge",
    "start_of_intro": "Intro",
    "start_of_outro": "Outro",
}


@dataclass(frozen=True)
class ChordSpan:
    name: str  # as written, without any *N length
    bars: float | None  # None = the song's default chord length
    col: int  # character column in the lyric it sits over


@dataclass(frozen=True)
class Line:
    section: str
    text: str  # lyric, "" for a chords-only line
    chords: tuple[ChordSpan, ...]


@dataclass(frozen=True)
class Song:
    title: str
    artist: str
    tempo: float  # beats per minute, as the sheet gives it
    beats_per_bar: int
    capo: int
    chord_bars: float | None  # length of a chord without *N; None = SongConfig.chord_bars
    lines: tuple[Line, ...]
    path: Path | None = None
    tempo_given: bool = True
    notes: tuple[str, ...] = field(default=())  # parse warnings, for the player to see

    @property
    def chord_names(self) -> tuple[str, ...]:
        """Distinct chord names in order of first use."""
        seen: dict[str, None] = {}
        for line in self.lines:
            for c in line.chords:
                seen.setdefault(c.name, None)
        return tuple(seen)


def load_sheet(path: str | Path) -> Song:
    p = Path(path)
    song = parse_sheet(p.read_text(encoding="utf-8"), default_title=p.stem)
    return Song(**{**song.__dict__, "path": p})


def parse_sheet(text: str, default_title: str = "Untitled") -> Song:
    meta: dict[str, str] = {}
    lines: list[Line] = []
    notes: list[str] = []
    section = ""
    pending: list[ChordSpan] | None = None  # a chord line waiting for its lyric

    def flush() -> None:
        nonlocal pending
        if pending is not None:
            lines.append(Line(section, "", tuple(pending)))
            pending = None

    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if d := _DIRECTIVE.match(stripped):
            flush()
            key, value = d.group(1).lower(), (d.group(2) or "").strip()
            if key in _SECTION_DIRECTIVES:
                section = value or _SECTION_DIRECTIVES[key]
            elif not key.startswith("end_of") and key not in ("eov", "eoc", "eob"):
                meta[key] = value
            continue
        if _is_section(stripped):
            flush()
            section = stripped[1:-1].strip()
            continue
        spans = _chord_line(line)
        if spans is not None:
            flush()
            pending = spans
            continue
        if _INLINE.search(line):
            flush()
            try:
                lines.append(_chordpro_line(section, line))
            except UnknownChordError as exc:
                notes.append(f"line {n}: {exc}")
            continue
        # A lyric: under a chord line it takes those chords, else it has none. Columns
        # count from the lyric's first character.
        lead = len(line) - len(line.lstrip())
        spans = tuple(ChordSpan(s.name, s.bars, max(0, s.col - lead)) for s in pending or ())
        lines.append(Line(section, stripped, spans))
        pending = None
    flush()

    if not any(line.chords for line in lines):
        raise ValueError("the sheet has no chords")
    tempo_given = "tempo" in meta
    tempo = _number(meta.get("tempo"), DEFAULT_TEMPO, "tempo", notes)
    if not tempo_given:
        notes.append(f"no {{tempo: ...}} line: using {DEFAULT_TEMPO:g} BPM")
    time_sig = meta.get("time", "4/4")
    try:
        beats_per_bar = int(time_sig.split("/")[0])
    except ValueError:
        beats_per_bar = 4
        notes.append(f"time {time_sig!r} not understood: using 4/4")
    return Song(
        title=meta.get("title") or meta.get("t") or default_title,
        artist=meta.get("artist") or meta.get("subtitle") or meta.get("st") or "",
        tempo=tempo,
        beats_per_bar=max(1, beats_per_bar),
        capo=int(_number(meta.get("capo"), 0, "capo", notes)),
        chord_bars=(
            _number(meta["chord_bars"], 1.0, "chord_bars", notes) if "chord_bars" in meta else None
        ),
        lines=tuple(lines),
        tempo_given=tempo_given,
        notes=tuple(notes),
    )


def _number(value: str | None, default: float, what: str, notes: list[str]) -> float:
    if value is None or value == "":
        return default
    try:
        x = float(value.split()[0])
    except ValueError:
        notes.append(f"{what} {value!r} not understood: using {default:g}")
        return default
    return x if x > 0 or what == "capo" else default


def _is_section(s: str) -> bool:
    """``[Verse 1]`` on its own line, as opposed to a ChordPro ``[Am]`` chord."""
    if not (s.startswith("[") and s.endswith("]")) or s.count("[") != 1:
        return False
    return _token_chord(s[1:-1].strip()) is None


def _token_chord(token: str) -> tuple[str, float | None] | None:
    """``"Am*2"`` -> ``("Am", 2.0)``; None if the token is not a chord."""
    bars = None
    if m := _LENGTH.match(token):
        token, bars = m.group(1), float(m.group(2))
    try:
        parse_chord(token)
    except UnknownChordError:
        return None
    return token, bars


def _chord_line(line: str) -> list[ChordSpan] | None:
    """A line of only chord names (``|`` bar marks allowed), else None."""
    spans = []
    for m in re.finditer(r"\S+", line):
        tok = m.group(0)
        if tok == "|":
            continue
        tok = tok.strip("[]")  # "[G] [D]" chord-only ChordPro lines read the same
        parsed = _token_chord(tok)
        if parsed is None:
            return None
        spans.append(ChordSpan(parsed[0], parsed[1], m.start()))
    return spans or None


def _chordpro_line(section: str, line: str) -> Line:
    text, spans, pos = [], [], 0
    for m in _INLINE.finditer(line):
        text.append(line[pos : m.start()])
        col = sum(len(t) for t in text)
        parsed = _token_chord(m.group(1).strip())
        if parsed is None:
            raise UnknownChordError(f"not a chord: [{m.group(1)}]")
        spans.append(ChordSpan(parsed[0], parsed[1], col))
        pos = m.end()
    text.append(line[pos:])
    lyric = "".join(text)
    lead = len(lyric) - len(lyric.lstrip())
    spans = [ChordSpan(s.name, s.bars, max(0, s.col - lead)) for s in spans]
    return Line(section, lyric.strip(), tuple(spans))
