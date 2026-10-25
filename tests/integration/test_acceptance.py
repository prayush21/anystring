"""tools/acceptance: measurements and verdicts on scripted sessions."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from anystring.config import Calibration
from anystring.pipeline import FrameInput, Pipeline
from anystring.replay.record import FrameRecord, Session, save_session
from anystring.types import NoteEvent
from tests.integration.song import frames
from tools.acceptance import Hit, Strum, group_strums, main, measure, report, verdicts

VOICED = {None: 6, "C": 5}


def song_session(cfg) -> Session:
    pipe = Pipeline(cfg)
    recs = []
    for f in frames(cfg)[0]:
        inp = FrameInput(f.t, f.t + 0.002, f.pose, f.hands, ())
        recs.append(FrameRecord(inp, tuple(pipe.step(inp))))
    return Session(cfg, Calibration(), recs, meta={"pose_source": "flow"})


def strums_at(times, directions=("down", "up")) -> list[Strum]:
    return [
        Strum(t, t + 0.02, directions[i % len(directions)], (0, 1, 2), 3, 3, "C")
        for i, t in enumerate(times)
    ]


def test_song_measures_every_strum_with_direction_and_chord(cfg):
    take = measure(song_session(cfg))
    assert take.replay_matches
    assert take.frames > 0 and take.fps == pytest.approx(60.0)
    assert take.dropped_frames == 0 and take.tracked == 1.0
    assert take.lag_ms[0] == pytest.approx(2.0)
    played = take.played
    assert [s.direction for s in played] == ["down", "up"] * 4
    assert [s.chord for s in played][::2] == ["C", "G", "Am", "F"]
    assert all(s.full for s in played)
    assert take.notes == sum(s.notes for s in played)
    # The open-hand sweep, then the open hand moving onto the head to mute.
    lift, to_mute = take.blocked
    assert (lift.direction, lift.strings) == ("down", (0, 1, 2, 3, 4, 5))
    assert to_mute.direction == "up" and to_mute.notes == 0
    assert take.damps > 0


def test_song_verdict_wants_the_chords_in_order(cfg):
    take = measure(song_session(cfg))
    (v,) = verdicts(take, "song")
    assert v.status == "CHECK" and "C G Am F" in v.measured
    shuffled = replace(take, strums=[replace(s, chord="G") for s in take.strums])
    assert verdicts(shuffled, "song")[0].status == "FAIL"


def test_group_strums_splits_on_direction_gap_and_string_order():
    down = [Hit(1.0 + 0.01 * i, i, "down", True, "C") for i in range(1, 6)]
    up = [Hit(1.1 + 0.01 * i, 5 - i, "up", True, "C") for i in range(6)]
    later = [Hit(1.5, 2, "up", True, "C"), Hit(1.51, 1, "up", False, "C")]
    restart = [Hit(1.52, 4, "up", True, "C")]  # higher string in an up sweep: new strum
    strums = group_strums(down + up + later + restart, VOICED)
    assert [(s.direction, s.strings) for s in strums] == [
        ("down", (1, 2, 3, 4, 5)),
        ("up", (5, 4, 3, 2, 1, 0)),
        ("up", (2, 1)),
        ("up", (4,)),
    ]
    assert [s.notes for s in strums] == [5, 6, 1, 1]
    assert strums[0].full and not strums[2].full


def test_group_strums_counts_a_fully_blocked_sweep_as_silent():
    hits = [Hit(2.0 + 0.01 * i, i, "down", False, None) for i in range(6)]
    (s,) = group_strums(hits, VOICED)
    assert not s.played and s.voiced == 6


def test_idle_verdict(cfg):
    take = replace(measure(song_session(cfg)), duration_s=125.0)
    assert verdicts(replace(take, strums=strums_at([5.0])), "idle")[0].status == "PASS"
    assert verdicts(replace(take, strums=strums_at([5.0, 9.0])), "idle")[0].status == "FAIL"
    short = replace(take, strums=[], duration_s=60.0)
    assert verdicts(short, "idle")[0].status == "CHECK"


def test_strums_verdict_caps_extras_and_counts_ups(cfg):
    take = measure(song_session(cfg))
    good = replace(take, strums=strums_at([0.3 * i for i in range(100)]))
    assert [v.status for v in verdicts(good, "strums")] == ["PASS", "PASS"]
    few_ups = replace(take, strums=strums_at([0.3 * i for i in range(100)], ("down", "down", "up")))
    detected, ups = verdicts(few_ups, "strums")
    assert detected.status == "FAIL" and "17 extra" in detected.measured
    assert ups.status == "FAIL" and "33/50" in ups.measured
    assert "alternation breaks" in ups.measured


def test_burst_verdict_needs_the_rate_and_no_drops(cfg):
    take = measure(song_session(cfg))
    steady = [i / 8.5 for i in range(100)]
    assert verdicts(replace(take, strums=strums_at(steady)), "burst")[0].status == "PASS"
    dropped = steady[:40] + steady[41:]
    (v,) = verdicts(replace(take, strums=strums_at(dropped)), "burst")
    assert v.status == "FAIL" and "1 drops" in v.measured
    slow = [i / 7.0 for i in range(100)]
    assert verdicts(replace(take, strums=strums_at(slow)), "burst")[0].status == "FAIL"


def test_move_verdict_never_passes_without_a_drift_check(cfg):
    take = measure(song_session(cfg))
    (moving,) = verdicts(replace(take, angle_range_deg=40.0), "move")
    assert moving.status == "CHECK" and "stayed on the head" in moving.measured
    lost = replace(take, angle_range_deg=40.0, tracked=0.97)
    assert verdicts(lost, "move")[0].status == "FAIL"
    (still,) = verdicts(replace(take, origin_travel_hl=0.1, angle_range_deg=5.0), "move")
    assert still.status == "CHECK" and "barely moved" in still.measured


def test_tracking_counts_from_the_first_lock(cfg):
    s = song_session(cfg)
    for i in range(30):  # half a second of clicking before the lock
        r = s.frames[i]
        pose = replace(r.inp.pose, visible=False)
        s.frames[i] = FrameRecord(replace(r.inp, pose=pose, hands=()), ())
    s.frames[60] = FrameRecord(
        replace(s.frames[60].inp, pose=replace(s.frames[60].inp.pose, visible=False)),
        s.frames[60].events,
    )
    take = measure(s)
    assert take.locked_after_s == pytest.approx(0.5)
    assert take.tracked == pytest.approx((len(s.frames) - 31) / (len(s.frames) - 30))
    assert "locked after 0.5 s" in report(take, [])


def test_report_says_when_the_racket_was_never_locked(cfg):
    s = song_session(cfg)
    s.frames = [
        FrameRecord(replace(r.inp, pose=replace(r.inp.pose, visible=False)), ()) for r in s.frames
    ]
    take = measure(s)
    assert take.locked_after_s is None and take.tracked == 0.0
    assert "never locked" in report(take, [])


def test_report_warns_when_the_replay_differs(cfg):
    s = song_session(cfg)
    first = next(i for i, r in enumerate(s.frames) if r.events)
    r = s.frames[first]
    s.frames[first] = replace(r, events=(*r.events, NoteEvent(r.inp.t, "on", 0, 40, 64)))
    take = measure(s)
    assert not take.replay_matches
    assert "WARNING: replay differs" in report(take, [])


def test_cli_prints_writes_json_and_fails_on_fail(cfg, tmp_path, capsys):
    s = song_session(cfg)
    s.meta["detect_to_audio_ms"] = {"p50": 30.0, "p95": 45.0, "n": 16}
    d = save_session(s, tmp_path / "song")
    out = tmp_path / "song.json"
    assert main([str(d), "--check", "song", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "[CHECK] four-chord song" in text and "detect-to-audio (app log) p50 30.0" in text
    data = json.loads(out.read_text())
    assert data["verdicts"][0]["status"] == "CHECK"
    assert sum(st["played"] for st in data["strums"]) == 8
    assert main([str(d), "--check", "idle"]) == 1  # 8 strums is not idle
