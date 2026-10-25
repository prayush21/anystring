"""Song mode through the live app: a voice request loads a sheet, the handle relabels,
"start" counts in and the song plays on; recordings replay exactly. Lyrics here are
made up."""

from __future__ import annotations

import numpy as np
import pytest

from racketguitar.app import LiveApp, main
from racketguitar.config import Calibration
from racketguitar.replay.harness import replay
from racketguitar.replay.record import load_session
from racketguitar.song.library import find_song
from racketguitar.song.player import SongPlayer
from racketguitar.song.voice import VoiceError
from racketguitar.ui.overlay import BGR, Overlay
from tests.integration.test_app import lock_song, song_devices

# Fast (so the scripted performance spans the whole song) and seven chords (so the
# four handle sections must relabel while it plays).
SHEET = """\
{title: Paper Boats}
{artist: The Placeholders}
{tempo: 240}
{chord_bars: 0.5}
[Verse]
Am      D       G
folding paper boats
C    C/B   Em
down the stream
[Chorus]
G  D  Em  C
Bm  F  Am  D
"""


@pytest.fixture
def song_dir(tmp_path):
    d = tmp_path / "songs"
    d.mkdir()
    (d / "paper-boats.txt").write_text(SHEET)
    return d


@pytest.fixture
def song_cfg(cfg):
    return cfg.with_overrides({"song": {"speed": 1.0, "count_in_beats": 4}})


def _app(cfg, song_dir, **kw):
    devices, audio = song_devices(cfg, **kw)
    app = LiveApp(cfg, Calibration(), devices, song_dir=song_dir)
    return app, devices, audio


def test_voice_request_relabels_then_start_counts_in_and_plays(song_cfg, song_dir):
    app, devices, _ = _app(song_cfg, song_dir)
    items = devices.camera._items
    lock_song(app, song_cfg)
    view = app.process(items.pop(0))
    assert view.song is None and view.snapshot.chord_names == ("C", "G", "Am", "F")

    app.handle_heard("Let's play paper boats by the placeholders")
    view = app.process(items.pop(0))
    st = view.song_state
    assert view.song is not None and view.song.song.title == "Paper Boats"
    assert st.phase == "ready" and st.next_chord == "Am"
    # The chords changed visibly: the handle now carries the song's first set.
    assert view.snapshot.chord_names == st.slots == ("Am", "D", "G", "C")
    assert "Paper Boats" in view.status.message

    app.handle_heard("start")
    phases, counts = [], []
    for frame in items:
        view = app.process(frame)
        st = view.song_state
        phases.append(st.phase)
        if st.count is not None:
            counts.append(st.count)
        # Every frame, the handle plays exactly the labels the song shows.
        assert view.snapshot.chord_names == st.slots
        assert view.snapshot.voicing == (
            dict(st.chords)[view.snapshot.chord]
            if view.snapshot.chord is not None
            else song_cfg.strings.tuning
        )
    assert phases[0] == "countin" and "playing" in phases
    assert sorted(set(counts), reverse=True) == [3, 2, 1, 0]
    assert len(app.player.plan) > 1  # seven chords on four sections: it relabelled


def test_song_take_records_relabels_and_replays_exactly(tmp_path, song_cfg, song_dir):
    holder: dict[str, LiveApp] = {}
    devices, audio = song_devices(song_cfg, on_done=lambda: holder["app"].stop_event.set())
    song = find_song("paper boats", song_dir).song
    app = LiveApp(song_cfg, Calibration(), devices, record_dir=tmp_path / "take", song=song)
    holder["app"] = app
    lock_song(app, song_cfg)
    app.command(app.toggle_song)
    app.run(display=False, seconds=60)

    session = load_session(tmp_path / "take")
    relabels = [f.inp.chords for f in session.frames if f.inp.chords is not None]
    assert len(relabels) >= 2  # the first set, then at least one relabel mid-song
    assert relabels[0][0][0] == "Am"
    assert any(e.kind == "on" for e in audio.events)
    result = replay(tmp_path / "take")
    assert result.matches, result.mismatches
    assert result.events == audio.events


def test_space_stops_and_restarts_and_x_leaves(song_cfg, song_dir):
    app, devices, _ = _app(song_cfg, song_dir)
    items = devices.camera._items
    app.toggle_song()
    assert "no song" in app._message
    app.handle_heard("play paper boats")
    app.process(items.pop(0))
    app.command(app.toggle_song)
    assert app.process(items.pop(0)).song_state.phase == "countin"
    app.command(app.toggle_song)
    assert app.process(items.pop(0)).song_state.phase == "ready"
    app.handle_heard("from the top")
    assert app.process(items.pop(0)).song_state.phase == "countin"
    app.handle_heard("stop")
    assert app.process(items.pop(0)).song_state.phase == "ready"
    app.command(app.leave_song)
    view = app.process(items.pop(0))
    assert view.song is None and view.snapshot.chord_names == ("C", "G", "Am", "F")


def test_unmatched_request_and_voice_errors_only_show_a_message(song_cfg, song_dir):
    app, devices, _ = _app(song_cfg, song_dir)
    items = devices.camera._items
    app.handle_heard("let's play bohemian rhapsody")
    view = app.process(items.pop(0))
    assert view.song is None and "no song file" in view.status.message

    def deaf() -> str:
        raise VoiceError("voice needs $ELEVENLABS_API_KEY")

    app.hear = deaf
    app._listening.set()
    app._listen_worker()  # what the worker thread runs
    assert not app._listening.is_set()
    assert "ELEVENLABS" in app.process(items.pop(0)).status.message

    app.hear = lambda: "paper boats"  # a bare title loads the song too
    app._listen_worker()
    view = app.process(items.pop(0))
    assert view.song is not None and view.song.song.title == "Paper Boats"


def test_overlay_draws_every_song_phase(song_cfg, song_dir):
    app, devices, _ = _app(song_cfg, song_dir)
    items = devices.camera._items
    lock_song(app, song_cfg)
    app.handle_heard("play paper boats")
    overlay = Overlay(song_cfg)
    view = app.process(items.pop(0))
    ready = overlay.draw(view)
    green = np.all(np.abs(ready.astype(int) - BGR["next"]) <= 10, axis=-1)
    assert green.sum() > 500  # the ready card's border and the next-section glow

    p: SongPlayer = view.song
    p.start(view.t)
    spb = p.seconds_per_beat
    for beat in (-3.5, -0.5, 0.1, 5.0, p.timeline.total_beats + 1.0):
        t = view.t + (beat + p.count_in) * spb
        st = p.state(t)
        out = overlay.draw(
            type(view)(view.image, t, view.pose, view.snapshot, view.status, song=p, song_state=st)
        )
        assert out.shape == view.image.shape


def test_song_command_prints_the_plan(song_dir, capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["song", str(song_dir / "paper-boats.txt"), "--no-lyrics", "--speed", "0.5"])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    assert "Paper Boats" in out and "120 BPM" in out and "folding" not in out
    assert "handle sections" in out
