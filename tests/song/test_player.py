from __future__ import annotations

import math

import pytest

from racketguitar.config import Config
from racketguitar.song.player import SongPlayer, build_timeline, plan_slots
from racketguitar.song.sheet import parse_sheet
from racketguitar.song.voicings import voicing

# Seven chords, so four handle sections must be relabelled during the song.
SEVEN = """\
{title: Seven}
{tempo: 120}
[Verse]
Am      D
words here
G       C   C/B
more words
Em  C
[Chorus]
G  D  Em  C
[Bridge]
Bm  F  Bm  F
Am  D
"""


def player(sheet: str = SEVEN, **song_cfg) -> SongPlayer:
    cfg = Config().with_overrides({"song": {"speed": 1.0, **song_cfg}})
    return SongPlayer(parse_sheet(sheet), cfg)


def test_timeline_lengths_merging_and_slash_simplifying():
    song = parse_sheet("{tempo: 60}\nG  C  C/B  D*2\nla la\n\nlyric with no chord\n")
    tl = build_timeline(song, beats_per_chord=4.0, simplify_slash=True)
    # C/B plays as C and merges with it; the lyric with no chord holds D one more bar.
    assert [(c.name, c.beat, c.beats) for c in tl.chords] == [
        ("G", 0.0, 4.0),
        ("C", 4.0, 8.0),
        ("D", 12.0, 12.0),
    ]
    assert [(ln.beat, ln.beats) for ln in tl.lines] == [(0.0, 20.0), (20.0, 4.0)]
    assert [name for _, name in tl.lines[0].labels] == ["G", "C", "D"]
    keep = build_timeline(song, beats_per_chord=4.0, simplify_slash=False)
    assert [c.name for c in keep.chords] == ["G", "C", "C/B", "D"]


def test_four_or_fewer_chords_never_relabel():
    p = player("G D Em C\nG D Em C\n")
    assert len(p.plan) == 1 and p.plan[0].slots == ("G", "D", "Em", "C")
    p3 = player("G D C\n")
    assert p3.plan[0].slots == ("G", "D", "C")  # three sections, one per chord


def test_relabel_plan_always_has_the_chord_in_time_and_never_moves_one():
    p = player()
    chords = p.timeline.chords
    assert len({c.name for c in chords}) == 7 and len(p.plan) > 1
    for c in chords:  # every chord is on the handle for its whole span
        for b in (c.beat, c.beat + c.beats / 2, c.end - 1e-6):
            assert c.name in p.slots_at(b), (c, b)
    for a, b in zip(p.plan, p.plan[1:], strict=False):  # a chord on the handle stays put
        for i, name in enumerate(a.slots):
            if name in b.slots:
                assert b.slots[i] == name
    beats = [c.beat for c in p.plan]
    assert beats == sorted(beats) and beats[0] == -math.inf


def test_plan_relabels_as_early_as_possible():
    chords = build_timeline(parse_sheet("A B C D A E\n"), 4.0, True).chords
    plan = plan_slots(chords, 4)
    # E replaces a chord not needed again; B was last needed at beat 8, so the swap
    # can happen there, long before E arrives at beat 20.
    assert plan[-1].beat == 8.0 and "E" in plan[-1].slots and "A" in plan[-1].slots


def test_phases_count_in_and_next_section():
    p = player(count_in_beats=4)
    spb = p.seconds_per_beat
    assert spb == pytest.approx(0.5)  # 120 BPM at speed 1.0
    ready = p.state(5.0)
    assert ready.phase == "ready" and ready.next_chord == "Am" and ready.chord is None
    assert ready.slots[ready.next_slot] == "Am"
    assert ready.chords == tuple((n, voicing(n, Config().strings.tuning)) for n in ready.slots)

    p.start(10.0)
    counts = [p.state(10.0 + k * spb + 0.01) for k in range(4)]
    assert [s.phase for s in counts] == ["countin"] * 4
    assert [s.count for s in counts] == [3, 2, 1, 0]

    first = p.state(10.0 + 4 * spb + 0.01)
    assert first.phase == "playing" and first.chord == "Am" and first.line == 0
    assert first.next_chord == "D" and first.slots[first.next_slot] == "D"
    assert first.beats_to_next == pytest.approx(4.0 - 0.02, abs=1e-6)

    end = 10.0 + (4 + p.timeline.total_beats) * spb
    done = p.state(end + 1.0)
    assert done.phase == "done" and done.next_chord is None and done.chord is None
    p.stop()
    assert p.state(end).phase == "ready" and not p.started


def test_next_slot_always_points_at_the_next_chord():
    p = player()
    p.start(0.0)
    spb = p.seconds_per_beat
    t = 0.0
    while (s := p.state(t)).phase != "done":
        if s.next_chord is not None:
            upcoming = p.slots_at(s.beat + s.beats_to_next)
            assert upcoming[s.next_slot] == s.next_chord
        if s.chord is not None:
            assert s.chord in s.slots  # what you should be playing is on the handle
        t += spb / 3


def test_speed_scales_tempo():
    assert player(speed=0.5).bpm == pytest.approx(60.0)


def test_unknown_chord_is_an_error():
    with pytest.raises(ValueError):
        player("G Hmaj\n")  # "Hmaj" is not a chord, so the line is a lyric: no chords
