"""The coach through the live app: feedback when a song ends and when asked, the take it
judges, and the same take rebuilt from a recording. No LLM or speech calls: phrasing is
the template and speaking is captured."""

from __future__ import annotations

import pytest

from anystring.app import LiveApp, main
from anystring.config import Calibration
from anystring.replay.take import log_from_session
from anystring.song.library import find_song
from tests.integration.test_app import lock_song, song_devices

# One beat per chord at 240 BPM: the song is over well before the scripted strumming.
SHEET = """\
{title: Short Song}
{tempo: 240}
{chord_bars: 0.25}
C G Am F
"""


@pytest.fixture
def song_dir(tmp_path):
    d = tmp_path / "songs"
    d.mkdir()
    (d / "short-song.txt").write_text(SHEET)
    return d


@pytest.fixture
def coach_cfg(cfg):
    return cfg.with_overrides(
        {"song": {"speed": 1.0, "count_in_beats": 1}, "coach": {"llm": False}}
    )


def _coached(app: LiveApp):
    """Run feedback inline and capture what it judged and said."""
    reports, said = [], []
    phrase = app.phrase_feedback
    app.run_worker = lambda fn: fn()
    app.wait_quiet = lambda: True
    app.say = said.append
    app.phrase_feedback = lambda r: (reports.append(r), phrase(r))[1]
    return reports, said


def test_song_end_gives_feedback_on_that_song(coach_cfg, song_dir):
    devices, _ = song_devices(coach_cfg)
    app = LiveApp(coach_cfg, Calibration(), devices, song_dir=song_dir)
    reports, said = _coached(app)
    lock_song(app, coach_cfg)
    app.handle_heard("let's play short song")
    app.handle_heard("start")
    for frame in devices.camera._items:
        app.process(frame)

    assert len(reports) == 1  # once, when the song reached its end
    r = reports[0]
    assert r.song is not None and r.song.title == "Short Song" and r.song.finished
    assert [c.name for c in r.song.chords] == ["C", "G", "Am", "F"]
    assert r.timing is not None and r.strokes.strokes >= 4
    assert said == [app.last_feedback] and app.last_feedback
    assert app.last_feedback in app._message  # shown on screen too


def test_asking_how_was_that_judges_the_free_playing(coach_cfg, song_dir):
    devices, _ = song_devices(coach_cfg)
    app = LiveApp(coach_cfg, Calibration(), devices, song_dir=song_dir)
    reports, said = _coached(app)
    lock_song(app, coach_cfg)
    items = devices.camera._items
    last = items.pop()
    for frame in items:
        app.process(frame)
    assert reports == []
    app.handle_heard("How was that?")
    app.process(last)

    (r,) = reports
    assert r.song is None and r.timing is None
    # Four chords strummed down and up, plus the open-hand sweep the gate blocked.
    assert r.strokes.strokes >= 8 and r.strokes.missed >= 1
    assert said == [app.last_feedback]


def test_feedback_waits_for_quiet_and_skips_speaking_if_never_quiet(coach_cfg, song_dir):
    devices, _ = song_devices(coach_cfg)
    app = LiveApp(coach_cfg, Calibration(), devices, song_dir=song_dir)
    _, said = _coached(app)
    app.wait_quiet = lambda: False
    lock_song(app, coach_cfg)
    for frame in devices.camera._items:
        app.process(frame)
    app.request_feedback()
    assert said == [] and app.last_feedback  # still shown, not spoken


def test_recorded_song_rebuilds_the_same_take_and_cli_coaches_it(
    tmp_path, coach_cfg, song_dir, capsys
):
    cfg = coach_cfg.with_overrides({"coach": {"speak": False}})
    holder: dict[str, LiveApp] = {}
    devices, _ = song_devices(cfg, on_done=lambda: holder["app"].stop_event.set())
    song = find_song("short song", song_dir).song
    app = LiveApp(cfg, Calibration(), devices, record_dir=tmp_path / "take", song=song)
    holder["app"] = app
    reports, _ = _coached(app)
    lock_song(app, cfg)
    app.command(app.toggle_song)
    app.run(display=False, seconds=60)

    assert len(reports) == 1
    log = log_from_session(tmp_path / "take")
    assert log.song is not None and log.song.title == "Short Song"
    live = app.takes.log(app._song_run.start)
    assert log.hits == live.hits and len(log.hits) > 0
    assert [f.chord for f in log.frames] == [f.chord for f in live.frames]

    with pytest.raises(SystemExit) as done:
        main(["coach", str(tmp_path / "take"), "--no-llm"])
    assert done.value.code == 0
    out = capsys.readouterr().out
    assert '"title": "Short Song"' in out and "\ncoach: " in out
