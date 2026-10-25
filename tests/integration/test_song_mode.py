"""Song mode through the live app: a voice request loads a sheet, the handle relabels,
"start" counts in and the song plays on; recordings replay exactly. Lyrics here are
made up."""

from __future__ import annotations

import numpy as np
import pytest

from anystring.app import LiveApp, main
from anystring.config import Calibration
from anystring.replay.harness import replay
from anystring.replay.record import load_session
from anystring.song.library import find_song
from anystring.song.player import SongPlayer
from anystring.song.voice import VoiceError
from anystring.ui.overlay import BGR, Overlay
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
    assert green.sum() > 500  # the start cue and the next-section glow

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


@pytest.mark.parametrize("mirror", [False, True])
def test_song_hud_leaves_lock_prompts_clear(song_cfg, song_dir, mirror):
    from dataclasses import replace

    app, devices, _ = _app(song_cfg, song_dir)
    app.handle_heard("play paper boats")
    view = app.process(devices.camera._items.pop(0))
    overlay = Overlay(song_cfg.with_overrides({"ui": {"mirror": mirror}}))
    prompt = "Click the TOP of the racket head"
    plain = overlay.draw(replace(view, song=None), prompt=prompt)
    with_song = overlay.draw(view, prompt=prompt)
    assert np.array_equal(with_song, plain)


def test_wrapped_lyrics_keep_every_word_and_source_column():
    import cv2

    from anystring.ui.song_hud import FONT, wrap_lyric

    text = "Folding paper boats with extraordinarilylongunbrokenwords down the stream"
    rows = wrap_lyric(text, 150, 0.72)
    assert "".join(s for _, s in rows).replace(" ", "") == text.replace(" ", "")
    for start, row in rows:
        assert text[start : start + len(row)] == row
        assert cv2.getTextSize(row, FONT, 0.72, 1)[0][0] <= 150


def test_song_hud_distinguishes_target_from_held_chord(song_cfg, song_dir, monkeypatch):
    from anystring.ui import song_hud

    player = SongPlayer(find_song("paper boats", song_dir).song, song_cfg)
    player.start(0)
    st = player.state(player.count_in * player.seconds_per_beat + 0.01)
    labels = []
    monkeypatch.setattr(song_hud, "_text", lambda img, text, *a, **kw: labels.append(text))
    img = np.zeros((720, 1280, 3), np.uint8)
    song_hud.draw_song_hud(img, player, st, "G", True)
    assert "Am" in labels and "D" in labels
    assert "You: G" in labels
    assert "Grip 1" in labels and "REC" in labels


@pytest.mark.parametrize("size", [(360, 640), (720, 1280), (1080, 1920)])
def test_song_hud_preserves_camera_dimensions_and_play_area(song_cfg, song_dir, size):
    from anystring.ui.song_hud import HUD_HEIGHT, HUD_WIDTH, draw_song_hud

    player = SongPlayer(find_song("paper boats", song_dir).song, song_cfg)
    player.start(0)
    original = np.full((*size, 3), 120, np.uint8)
    for beat in (-3.5, 0.1, player.timeline.total_beats + 1):
        img = original.copy()
        state = player.state((beat + player.count_in) * player.seconds_per_beat)
        draw_song_hud(img, player, state, "G", True)
        assert img.shape == original.shape
        boundary = round(size[1] * HUD_HEIGHT / HUD_WIDTH)
        assert np.array_equal(img[boundary:], original[boundary:])
        assert not np.array_equal(img[:boundary], original[:boundary])


def test_lyrics_rise_and_gain_focus_at_their_timed_boundary(song_cfg, song_dir):
    from anystring.ui.song_hud import LYRIC_FOCUS_Y, _lyric_positions

    player = SongPlayer(find_song("paper boats", song_dir).song, song_cfg)
    boundary = player.timeline.lines[1].beat
    stages = []
    for seconds in (0, 0.1, 0.2, 0.4):
        positions = _lyric_positions(player, boundary + seconds / player.seconds_per_beat)
        stages.append(next((y, focus) for b, y, focus in positions if b.beat == boundary))
    assert stages[0][0] > LYRIC_FOCUS_Y and stages[0][1] == 0
    assert all(a[0] > b[0] and a[1] < b[1] for a, b in zip(stages, stages[1:], strict=False))
    assert stages[-1] == pytest.approx((LYRIC_FOCUS_Y, 1))
    # The boundary itself must not teleport the upcoming line to its new position.
    before = _lyric_positions(player, boundary - 1e-7)
    y_before = next(y for b, y, _ in before if b.beat == boundary)
    assert y_before == pytest.approx(stages[0][0])


def test_lyric_scroll_is_repeatable_after_seek_and_does_not_cover_controls(song_cfg, song_dir):
    from anystring.ui.song_hud import (
        LYRIC_HEIGHT,
        LYRIC_WIDTH,
        LYRIC_X,
        LYRIC_Y,
        _draw_scrolling_lyrics,
    )

    player = SongPlayer(find_song("paper boats", song_dir).song, song_cfg)
    player.start(0)
    background = np.full((224, 1280, 3), 70, np.uint8)

    def render(beat):
        img = background.copy()
        state = player.state((beat + player.count_in) * player.seconds_per_beat)
        _draw_scrolling_lyrics(img, player, state)
        return img

    beat = player.timeline.lines[1].beat + 0.2 / player.seconds_per_beat
    first = render(beat)
    render(beat + 8)
    assert np.array_equal(first, render(beat))
    outside = np.ones(first.shape[:2], dtype=bool)
    outside[LYRIC_Y : LYRIC_Y + LYRIC_HEIGHT, LYRIC_X : LYRIC_X + LYRIC_WIDTH] = False
    assert np.array_equal(first[outside], background[outside])


def test_long_lyrics_scroll_all_words_in_timed_pages():
    from anystring.song.player import TimedLine
    from anystring.ui.song_hud import _lyric_blocks

    words = " ".join(f"word{i}" for i in range(90))
    blocks = _lyric_blocks((TimedLine("Verse", words, 0, 16, ()),))
    assert len(blocks) > 1
    assert " ".join(row for block in blocks for row in block.rows) == words
    assert blocks[0].beat == 0
    assert blocks[-1].beat + blocks[-1].beats == pytest.approx(16)


def test_song_command_prints_the_plan(song_dir, capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["song", str(song_dir / "paper-boats.txt"), "--no-lyrics", "--speed", "0.5"])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    assert "Paper Boats" in out and "120 BPM" in out and "folding" not in out
    assert "handle sections" in out
