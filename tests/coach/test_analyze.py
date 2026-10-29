"""The take analysis measures what the coach talks about, from hand-written takes."""

from __future__ import annotations

import json

import pytest

from anystring.coach.analyze import analyze, facts, strokes_of
from anystring.config import CoachConfig
from tests.coach.takes import T0, beat_strokes, frames, held_as_due, song, stroke, take

CHORDS = ["C", "G", "Am", "F", "C", "G", "Am", "F"]


@pytest.fixture
def ccfg() -> CoachConfig:
    return CoachConfig()


def _song_take(offset=0.0, delays=None, chord_at=None):
    info = song(CHORDS)
    held = chord_at or held_as_due(info, delays)
    return take(beat_strokes(info, offset, held), frames(T0 - 1, info.chords[-1].end, held), info)


def test_strokes_group_hits_by_direction_and_gap(ccfg):
    hits = stroke(1.0, "down") + stroke(1.25, "up", strings=2) + stroke(1.3, "up", gated=6)
    strokes = strokes_of(tuple(hits), ccfg.stroke_gap_ms)
    assert [(s.direction, s.strings, s.played, s.gated) for s in strokes] == [
        ("down", 6, 6, 0),
        ("up", 6, 2, 6),  # 50 ms after the thin up-stroke: the same stroke
    ]


def test_clean_take_has_no_findings_and_praises(ccfg):
    r = analyze(_song_take(), ccfg)
    assert r.enough and r.timing is not None and r.tempo is None
    assert r.timing.on_time_share == 1.0 and abs(r.timing.mean_ms) < 1.0
    assert r.chord_accuracy == 1.0
    assert all(c.late == 0 and c.missed == 0 for c in r.changes)
    assert r.findings == ()
    assert any("timing was tight" in s for s in r.strengths)
    assert any("on time" in s for s in r.strengths)


@pytest.mark.parametrize(("offset", "key"), [(0.09, "drag"), (-0.08, "rush")])
def test_late_or_early_strums_are_dragging_or_rushing(ccfg, offset, key):
    r = analyze(_song_take(offset), ccfg)
    assert r.timing.mean_ms == pytest.approx(offset * 1000, abs=1.0)
    assert r.findings[0].key == key
    assert ("behind" if offset > 0 else "ahead of") in r.findings[0].fact


def test_slowing_down_through_the_song_is_drift(ccfg):
    info = song(CHORDS)
    end = info.chords[-1].end
    hits = []
    spb = info.seconds_per_beat
    n = int(round((end - info.t_beat0) / (spb / 2)))
    for k in range(n):
        t = info.t_beat0 + k * spb / 2 + 0.12 * k / n  # 0 -> 120 ms late
        hits += stroke(t, "down" if k % 2 == 0 else "up", chord=held_as_due(info)(t))
    r = analyze(take(hits, frames(T0 - 1, end, held_as_due(info)), info), ccfg)
    drift = next(f for f in r.findings if f.key == "drift")
    assert "fell behind" in drift.fact


def test_late_change_to_one_chord_is_named(ccfg):
    r = analyze(_song_take(delays={"Am": 0.3}), ccfg)
    worst = r.changes[0]
    assert (worst.frm, worst.to, worst.count, worst.late, worst.missed) == ("G", "Am", 2, 2, 0)
    assert worst.mean_lag_ms == pytest.approx(300, abs=20)
    # The down-strum on the beat and the up-strum after it still hit G.
    assert worst.wrong_strums == 4
    assert all(c.late == 0 for c in r.changes[1:])
    top = r.findings[0]
    assert top.key == "change:G>Am" and "G to Am" in top.fact and "late 2 of 2" in top.fact
    assert r.chord_accuracy < 1.0


def test_never_reaching_a_chord_is_missed(ccfg):
    info = song(CHORDS)
    base = held_as_due(info)
    r = analyze(_song_take(chord_at=lambda t: "G" if base(t) == "Am" else base(t)), ccfg)
    worst = r.changes[0]
    assert (worst.frm, worst.to, worst.missed) == ("G", "Am", 2)
    assert "never got there" in r.findings[0].fact


def test_letting_go_of_a_chord_early_is_rushing_the_change(ccfg):
    info = song(CHORDS)
    base = held_as_due(info)

    def held(t):  # off C 0.5 s before each change to G, on G on time
        nxt = next((c for c in info.chords if c.t > t), None)
        if base(t) == "C" and nxt is not None and nxt.name == "G" and nxt.t - t < 0.5:
            return None
        return base(t)

    r = analyze(_song_take(chord_at=held), ccfg)
    c_to_g = next(c for c in r.changes if (c.frm, c.to) == ("C", "G"))
    assert c_to_g.left_early == 2 and c_to_g.late == 0
    assert c_to_g.mean_left_early_ms == pytest.approx(500, abs=20)
    assert any(f.key == "rush_change:C>G" and "let go of C" in f.fact for f in r.findings)


def test_grip_missed_and_dropped_strokes(ccfg):
    hits = []
    for k in range(12):
        t = 1.0 + 0.25 * k
        gated = 6 if k in (2, 5, 8) else 3 if k in (4, 9) else 0
        hits += stroke(t, "down" if k % 2 == 0 else "up", gated=gated)
    r = analyze(take(hits, frames(0.5, 4.5)), ccfg)
    assert (r.strokes.missed, r.strokes.grip_drops) == (3, 2)
    keys = [f.key for f in r.findings]
    assert keys[:2] == ["missed", "grip_drop"]
    assert "pick grip" in r.findings[0].fact


def test_weak_up_strums_and_double_hits(ccfg):
    hits = []
    for k in range(12):
        down = k % 2 == 0
        hits += stroke(1.0 + 0.25 * k, "down" if down else "up", strings=6 if down else 2)
    blocked = frames(0.5, 4.5, blocked_at=lambda t: int((t - 0.5) * 2))  # one every 0.5 s
    r = analyze(take(hits, blocked), ccfg)
    assert r.strokes.up_strings == 2 and r.strokes.down_strings == 6
    assert r.strokes.double_hits_blocked == 8
    assert {"weak_up", "double_hits", "thin"} <= {f.key for f in r.findings}


def test_mute_followed_at_once_by_strumming_is_likely_unintended(ccfg):
    hits = []
    for k in range(10):
        hits += stroke(1.0 + 0.5 * k, "down")
    # Mutes fire 0.2 s before strokes 2, 4 and 6; one more long before the end.
    mutes = [1.8, 2.8, 3.8]
    r = analyze(
        take(hits, frames(0.5, 6.0, muted_at=lambda t: any(m <= t < m + 0.1 for m in mutes))),
        ccfg,
    )
    assert (r.mutes.mutes, r.mutes.likely_unintended) == (3, 3)
    assert any(f.key == "mute" for f in r.findings)


def test_free_play_tempo_drift_without_a_song(ccfg):
    hits, t = [], 1.0
    for k in range(18):
        hits += stroke(t, "down" if k % 2 == 0 else "up")
        t += 0.5 - 0.015 * k  # speeding up
    r = analyze(take(hits, frames(0.5, t)), ccfg)
    assert r.timing is None and r.tempo is not None
    assert r.tempo.end_per_min > r.tempo.start_per_min
    assert any(f.key == "tempo_drift" and "sped up" in f.fact for f in r.findings)


def test_too_few_strokes_is_not_enough(ccfg):
    r = analyze(take(stroke(1.0) + stroke(1.3, "up"), frames(0.5, 2.0)), ccfg)
    assert not r.enough and r.findings == ()


def test_lost_racket_is_reported(ccfg):
    hits = [h for k in range(8) for h in stroke(1.0 + 0.3 * k)]
    r = analyze(take(hits, frames(0.5, 4.0, visible_at=lambda t: t < 3.0)), ccfg)
    assert r.racket_lost_share == pytest.approx(1 / 3.5, abs=0.02)
    assert any(f.key == "lost" for f in r.findings)


def test_facts_are_compact_json(ccfg):
    r = analyze(_song_take(delays={"Am": 0.3}), ccfg)
    data = facts(r)
    text = json.dumps(data)
    assert len(text) < 3000
    assert data["song"]["title"] == "Test Song"
    assert data["chord_changes_worst_first"][0]["to"] == "Am"
    assert data["issues_ranked"][0]["fact"].startswith("the change from G to Am")
