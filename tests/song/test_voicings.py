from __future__ import annotations

import pytest

from anystring.config import STANDARD_TUNING
from anystring.song.voicings import (
    ALIASES,
    INTERVALS,
    UnknownChordError,
    chord_tones,
    is_chord,
    parse_chord,
    shape_to_midi,
    voicing,
)

ROOTS = ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def midi(shape: str, capo: int = 0):
    return shape_to_midi(shape, STANDARD_TUNING, capo)


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("C", "x32010"),
        ("G", "320003"),
        ("Am", "x02210"),
        ("Em", "022000"),
        ("D", "xx0232"),
        ("F", "133211"),
        ("Bm", "x24432"),  # A-form barre at 2
        ("F#m", "244222"),  # E-form barre at 2
        ("Cm", "x35543"),
        ("D/F#", "2x0232"),  # slash bass on the low E
        ("C/B", "x22010"),  # slash bass on the A string, low E muted
        ("G/B", "x20003"),
        ("A7sus4", "x02030"),
        ("Dsus4", "xx0233"),
    ],
)
def test_known_shapes(name, shape):
    assert voicing(name, STANDARD_TUNING) == midi(shape)


def test_capo_raises_every_note():
    assert voicing("Am", STANDARD_TUNING, capo=2) == midi("x02210", capo=2)
    assert midi("x02210", capo=2)[1] == 45 + 2


@pytest.mark.parametrize("root", ROOTS)
@pytest.mark.parametrize("quality", sorted(INTERVALS))
def test_every_voicing_sounds_only_chord_tones(root, quality):
    spelling = next(k for k, v in ALIASES.items() if v == quality)
    chord = parse_chord(root + spelling)
    notes = [n for n in voicing(chord, STANDARD_TUNING) if n is not None]
    pcs = {n % 12 for n in notes}
    assert pcs <= chord_tones(chord), (chord.name, notes)
    assert chord.root in pcs
    assert len(notes) >= (2 if quality == "5" else 3)
    played = zip(voicing(chord, STANDARD_TUNING), STANDARD_TUNING, strict=True)
    frets = [n - o for n, o in played if n is not None]
    assert 0 <= min(frets) and max(frets) <= 15  # on the neck


@pytest.mark.parametrize("bass", ["E", "F#", "G", "A", "B", "C"])
def test_slash_chords_keep_chord_tones_and_put_any_bass_lowest(bass):
    chord = parse_chord(f"C/{bass}")
    notes = [n for n in voicing(chord, STANDARD_TUNING) if n is not None]
    assert {n % 12 for n in notes} <= chord_tones(chord)
    if bass in ("B", "G", "E"):  # reachable from the open C shape
        assert min(notes) % 12 == parse_chord(f"C/{bass}").bass


def test_parse_and_aliases():
    assert parse_chord("Bbmaj7").root == 10 and parse_chord("Bbmaj7").quality == "maj7"
    assert parse_chord("Asus").quality == "sus4"
    assert parse_chord("Em9").quality == "m7"  # extension falls back to its core
    assert parse_chord("C/B").base_name == "C"
    assert is_chord("D/F#") and is_chord("Cadd9")
    for bad in ("H", "Verse", "Cxyz", "", "Am7/"):
        assert not is_chord(bad)
    with pytest.raises(UnknownChordError):
        parse_chord("Folding")


def test_needs_standard_tuning():
    with pytest.raises(ValueError):
        voicing("C", (40, 45, 50, 55, 59))
    with pytest.raises(ValueError):
        voicing("C", (38, 45, 50, 55, 59, 64))  # drop D
