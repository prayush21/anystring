from __future__ import annotations

from pathlib import Path

import pytest

from anystring.song.sheet import load_sheet

EXAMPLES = sorted((Path(__file__).parents[2] / "examples" / "songs").glob("*.txt"))


def test_there_are_example_sheets():
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_sheet_parses_cleanly_and_holds_no_lyrics(path):
    song = load_sheet(path)
    assert song.notes == ()
    assert song.tempo_given
    assert all(line.text == "" for line in song.lines)  # the repo is public: chords only
