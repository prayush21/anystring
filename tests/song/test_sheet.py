"""Sheet parsing. Every lyric here is made up, so no song text sits in the repo."""

from __future__ import annotations

import pytest

from racketguitar.song.sheet import DEFAULT_TEMPO, load_sheet, parse_sheet

ABOVE = """\
{title: Paper Boats}
{artist: The Placeholders}
{tempo: 120}
{capo: 2}

[Verse 1]
Am           D
Folding paper boats
G              C    C/B
Down the gutter stream

[Chorus]
G    D  Em  C
Sail away (sail away)
"""

CHORDPRO = """\
{title: Tin Roof}
{time: 3/4}
{chord_bars: 0.5}
{start_of_verse}
[G]Rain upon the [D*2]tin roof
[Em]all night [C]long
{end_of_verse}
{start_of_chorus: Refrain}
[G] [D] [Am*2]
{end_of_chorus}
"""


def test_chords_above_lyrics():
    song = parse_sheet(ABOVE)
    assert (song.title, song.artist, song.tempo, song.capo) == (
        "Paper Boats",
        "The Placeholders",
        120.0,
        2,
    )
    assert song.beats_per_bar == 4 and song.chord_bars is None and song.notes == ()
    first, second, chorus = song.lines
    assert first.section == "Verse 1" and first.text == "Folding paper boats"
    assert [(c.name, c.col) for c in first.chords] == [("Am", 0), ("D", 13)]
    assert [c.name for c in second.chords] == ["G", "C", "C/B"]
    assert chorus.section == "Chorus" and [c.name for c in chorus.chords] == ["G", "D", "Em", "C"]
    assert song.chord_names == ("Am", "D", "G", "C", "C/B", "Em")


def test_chordpro_inline_lengths_and_sections():
    song = parse_sheet(CHORDPRO)
    assert song.beats_per_bar == 3 and song.chord_bars == 0.5
    assert song.tempo == DEFAULT_TEMPO and not song.tempo_given and song.notes  # warned
    verse = song.lines[0]
    assert verse.section == "Verse" and verse.text == "Rain upon the tin roof"
    assert [(c.name, c.bars, c.col) for c in verse.chords] == [("G", None, 0), ("D", 2.0, 14)]
    refrain = song.lines[2]
    assert refrain.section == "Refrain" and refrain.text == ""
    assert [(c.name, c.bars) for c in refrain.chords] == [("G", None), ("D", None), ("Am", 2.0)]


def test_lyric_columns_follow_leading_spaces():
    song = parse_sheet("     Am     D\n  I said hello\n")
    (line,) = song.lines
    assert line.text == "I said hello"
    assert [(c.name, c.col) for c in line.chords] == [("Am", 3), ("D", 10)]


def test_lines_without_chords_and_chord_only_lines():
    song = parse_sheet("G   D\n\nA lone lyric\nEm\n")
    assert [(ln.text, [c.name for c in ln.chords]) for ln in song.lines] == [
        ("", ["G", "D"]),
        ("A lone lyric", []),
        ("", ["Em"]),
    ]


def test_section_header_is_not_a_chord():
    song = parse_sheet("[Intro]\n[Am]\n")
    assert song.lines[0].section == "Intro"
    assert [c.name for c in song.lines[0].chords] == ["Am"]


def test_errors_and_warnings():
    with pytest.raises(ValueError, match="no chords"):
        parse_sheet("{title: Silence}\nJust words here\n")
    song = parse_sheet("{tempo: fast}\n{time: waltz}\nG\nla [Hmm]la\n")
    assert song.tempo == DEFAULT_TEMPO and song.beats_per_bar == 4
    assert len(song.notes) == 3  # tempo, time, and the bad inline chord


def test_load_sheet_uses_file_name_as_fallback_title(tmp_path):
    p = tmp_path / "rainy-day.txt"
    p.write_text("G D\nla la\n")
    song = load_sheet(p)
    assert song.title == "rainy-day" and song.path == p
