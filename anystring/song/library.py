"""The player's song sheets, and matching a spoken or typed title to one of them.

Sheets live in ``<home>/songs/`` (``~/.anystring/songs``), outside the repo, since
they hold lyrics. Matching is fuzzy so speech-to-text slips still land: "take me on by
aha" finds "Take On Me" by a-ha.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from anystring.config import home_dir
from anystring.song.sheet import SHEET_SUFFIXES, Song, load_sheet

__all__ = ["songs_dir", "sheet_paths", "Match", "match_score", "find_song", "normalize"]


def songs_dir() -> Path:
    return home_dir() / "songs"


def sheet_paths(directory: Path | None = None) -> list[Path]:
    d = directory or songs_dir()
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in SHEET_SUFFIXES)


def normalize(text: str) -> str:
    """Lower case, hyphens and apostrophes dropped (a-ha -> aha), other punctuation
    to spaces, spaces collapsed."""
    text = re.sub(r"['’\-]", "", text.lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def match_score(query: str, song: Song) -> float:
    """0..1: how well ``query`` names ``song`` (by title, title + artist, or file name)."""
    q = normalize(query)
    if not q:
        return 0.0
    title, artist = normalize(song.title), normalize(song.artist)
    names = {title, f"{title} {artist}", f"{title} by {artist}"}
    if song.path is not None:
        names.add(normalize(song.path.stem))
    best = 0.0
    for name in names - {""}:
        seq = SequenceMatcher(None, q, name).ratio()
        words_q, words_n = set(q.split()), set(name.split())
        overlap = len(words_q & words_n) / len(words_q | words_n)
        best = max(best, seq, overlap)
    return best


@dataclass(frozen=True)
class Match:
    song: Song
    score: float


def find_song(query: str, directory: Path | None = None, min_score: float = 0.5) -> Match | None:
    """The best-matching sheet in ``directory``, or None if none reaches ``min_score``.
    A query that is a path to a sheet file loads that file."""
    p = Path(query).expanduser()
    if p.suffix.lower() in SHEET_SUFFIXES and p.is_file():
        return Match(load_sheet(p), 1.0)
    best: Match | None = None
    for path in sheet_paths(directory):
        try:
            song = load_sheet(path)
        except (ValueError, OSError):
            continue
        score = match_score(query, song)
        if best is None or score > best.score:
            best = Match(song, score)
    return best if best is not None and best.score >= min_score else None
