"""F8 clips: the rolling buffer, the saved file, the card's animation, and the app's wiring."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading

import numpy as np
import pytest
from scipy.io import wavfile

from anystring.app import LiveApp, clip_target, is_snap_key
from anystring.config import Calibration, load_config
from anystring.replay import clip as clip_module
from anystring.replay.clip import ClipBuffer
from anystring.replay.demo import DemoError
from anystring.types import NoteEvent
from anystring.ui import clip_hud
from anystring.ui.clip_hud import ClipStatus, card_visibility, draw_clip_hud
from tests.integration.song import frames
from tests.integration.test_app import _song_images, lock_song, song_devices

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
T0 = 100.0


def window(i: int) -> np.ndarray:
    return np.full((90, 160, 3), (i * 3) % 256, np.uint8)


def short_cfg(cfg, seconds: float = 2.0):
    return cfg.with_overrides({"clip": {"seconds": seconds}})


def feed(buf: ClipBuffer, seconds: float, fps: int = 30) -> None:
    for i in range(round(seconds * fps) + 1):
        buf.add_frame(T0 + i / fps, window(i))


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


# -- the rolling buffer -------------------------------------------------------------------


def test_buffer_keeps_only_the_last_seconds(cfg):
    buf = ClipBuffer(short_cfg(cfg, 2.0))
    buf.add_events([NoteEvent(T0 + 0.1, "on", 0, 40, 100), NoteEvent(T0 + 4.5, "on", 1, 45, 100)])
    feed(buf, 5.0)
    assert buf.buffered_s == pytest.approx(2.0, abs=0.05)
    assert len(buf._frames) == 61
    assert [e.t_detect for e in buf._events] == [T0 + 4.5]


def test_events_are_trimmed_without_any_frames(cfg):
    buf = ClipBuffer(short_cfg(cfg, 2.0))  # --no-display: events only
    for i in range(10):
        buf.add_events([NoteEvent(T0 + i, "on", 0, 40, 100)])
    assert len(buf._events) == 3


def test_nothing_buffered_means_nothing_saved(cfg, tmp_path):
    buf = ClipBuffer(cfg)
    assert buf.snap(tmp_path) is None and not buf.saving and buf.current is None


def test_clip_key_is_f8(cfg):
    assert is_snap_key(0xF70B) and is_snap_key(0xFFC5) and is_snap_key(ord("f"))
    assert not any(is_snap_key(k) for k in (-1, ord("q"), 27, 0xF70A))


def test_clip_folder_defaults_to_memories_under_the_repo(cfg, tmp_path):
    assert clip_target(cfg).name == "memories" and clip_target(cfg).is_absolute()
    absolute = cfg.with_overrides({"clip": {"folder": str(tmp_path / "mine")}})
    assert clip_target(absolute) == tmp_path / "mine"


def test_clip_settings_load_from_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[clip]\nseconds = 30\nfolder = "clips"\njpeg_quality = 60\n')
    c = load_config(path).clip
    assert (c.seconds, c.folder, c.jpeg_quality) == (30.0, "clips", 60)


# -- the saved file -----------------------------------------------------------------------


@needs_ffmpeg
def test_snap_writes_an_mp4_with_picture_and_sound(cfg, tmp_path):
    buf = ClipBuffer(short_cfg(cfg, 2.0))
    buf.add_events(
        [
            NoteEvent(T0 - 3.0, "on", 0, 40, 127),  # before the first frame: left out
            NoteEvent(T0 + 4.0, "on", 1, 45, 100),
        ]
    )
    feed(buf, 6.0)  # only the last 2 s are kept: T0+4 .. T0+6
    save = buf.snap(tmp_path / "memories")
    assert save is not None and save.path.parent == tmp_path / "memories"
    assert buf.wait(60)
    assert save.error is None and save.finished is not None and save.progress == 1.0
    assert save.done == save.frames == 61
    info = probe(save.path)
    assert {s["codec_type"] for s in info["streams"]} == {"video", "audio"}
    assert float(info["format"]["duration"]) == pytest.approx(2.0, abs=0.2)
    # the note inside the window is heard from the start; the one before it is not
    wav = tmp_path / "heard.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(save.path), str(wav)], check=True
    )
    rate, audio = wavfile.read(wav)
    assert np.abs(audio[: rate // 2]).max() > 0.01 * np.iinfo(audio.dtype).max
    assert not list(save.path.parent.glob("*.video.mp4"))


@needs_ffmpeg
def test_two_clips_in_one_second_do_not_overwrite(cfg, tmp_path):
    buf = ClipBuffer(short_cfg(cfg, 1.0))
    feed(buf, 1.0)
    first = buf.snap(tmp_path)
    assert buf.wait(60)
    second = buf.snap(tmp_path)
    assert buf.wait(60)
    assert first is not None and second is not None and first.path != second.path
    assert first.path.exists() and second.path.exists()


def test_a_failed_save_is_reported_not_raised(cfg, tmp_path, monkeypatch):
    class Broken:
        def __init__(self, *args, **kwargs) -> None:
            raise OSError("disk full")

    monkeypatch.setattr(clip_module, "ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(clip_module, "DemoRecorder", Broken)
    buf = ClipBuffer(cfg)
    feed(buf, 1.0)
    save = buf.snap(tmp_path)
    assert save is not None and buf.wait(10)
    assert save.error == "disk full" and save.finished is not None
    assert not list(tmp_path.glob("*.mp4"))


def test_missing_ffmpeg_is_explained(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("anystring.replay.demo.shutil.which", lambda name: None)
    buf = ClipBuffer(cfg)
    feed(buf, 1.0)
    with pytest.raises(DemoError, match="brew install ffmpeg"):
        buf.snap(tmp_path)


# -- the card -----------------------------------------------------------------------------


def status(phase="saving", finished=None, **kw) -> ClipStatus:
    return ClipStatus(phase, 10.0, finished, folder="memories", name="clip-1.mp4", **kw)


def test_card_slides_in_holds_and_leaves():
    saved = status("saved", finished=14.0, seconds=90.0)
    assert card_visibility(None, 11.0) == 0.0
    assert card_visibility(saved, 9.0) == 0.0  # before the press
    ramp = [card_visibility(saved, 10.0 + i * 0.05) for i in range(8)]
    assert ramp[0] == 0.0 and ramp[-1] == 1.0 and ramp == sorted(ramp)  # in, without a jump
    assert card_visibility(saved, 14.0 + clip_hud.HOLD_SAVED_S - 0.01) == 1.0  # held
    leave = [card_visibility(saved, 14.0 + clip_hud.HOLD_SAVED_S + i * 0.05) for i in range(8)]
    assert leave[0] == 1.0 and leave[-1] == 0.0 and leave == sorted(leave, reverse=True)
    # still saving: it stays as long as the save does
    assert card_visibility(status(), 500.0) == 1.0


def _canvas() -> np.ndarray:
    return np.full((720, 1280, 3), 90, np.uint8)


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.any(a != b, axis=-1)


def test_hint_sits_at_the_right_edge_and_is_quiet_when_idle():
    img = _canvas()
    draw_clip_hud(img, None, 5.0)
    ys, xs = np.nonzero(_changed(img, _canvas()))
    assert xs.min() > 1280 - 200 and xs.max() < 1280 - clip_hud.MARGIN + 2
    assert ys.min() >= clip_hud.TOP - 2 and ys.max() < clip_hud.TOP + 40  # a small pill
    assert img[clip_hud.TOP : clip_hud.TOP + 40, 1280 - 200 :].max() > 150  # the key cap shows


def test_press_flashes_then_the_card_replaces_the_hint():
    flash = _canvas()
    draw_clip_hud(flash, status(), 10.02)
    assert flash[:200, :200].mean() > 90 + 20  # whole window briefly whitened
    calm = _canvas()
    draw_clip_hud(calm, status(), 10.6)
    assert calm[:200, :200].mean() == 90
    amber = (calm == np.array(clip_hud.AMBER)).all(axis=-1)
    assert amber.any(), "saving arc and progress bar"


def test_progress_bar_grows_with_progress():
    def fill(p: float) -> int:
        img = _canvas()
        draw_clip_hud(img, status(progress=p, seconds=90.0), 11.0)
        return int((img == np.array(clip_hud.AMBER)).all(axis=-1)[: clip_hud.TOP + 80].sum())

    assert fill(0.0) < fill(0.5) < fill(0.95)


def test_saved_and_failed_cards_change_colour():
    saved, failed = _canvas(), _canvas()
    draw_clip_hud(saved, status("saved", finished=11.0, seconds=90.0), 12.0)
    draw_clip_hud(failed, status("failed", finished=11.0, error="disk full"), 12.0)
    assert (saved == np.array(clip_hud.GREEN)).all(axis=-1).any()
    assert (failed == np.array(clip_hud.RED)).all(axis=-1).any()
    assert not (failed == np.array(clip_hud.GREEN)).all(axis=-1).any()


def test_check_mark_draws_itself():
    def ink(since: float) -> int:
        img = _canvas()
        draw_clip_hud(img, status("saved", finished=11.0, seconds=90.0), 11.0 + since)
        return int((img == np.array(clip_hud.KEYCAP_TEXT)).all(axis=-1).sum())

    assert ink(0.0) < ink(0.1) < ink(0.5)


def test_card_stays_inside_small_windows():
    img = np.full((240, 320, 3), 90, np.uint8)
    draw_clip_hud(img, status("saved", finished=11.0, seconds=90.0), 12.0)  # no exception
    assert img.shape == (240, 320, 3)


# -- the app ------------------------------------------------------------------------------


@needs_ffmpeg
def test_f8_in_a_live_session_saves_the_window_and_the_notes(cfg, tmp_path):
    cfg = cfg.with_overrides({"clip": {"folder": str(tmp_path / "memories")}})
    done = threading.Event()
    devices, _ = song_devices(cfg, on_done=done.set)
    app = LiveApp(cfg, Calibration(), devices)
    lock_song(app, cfg)
    worker = threading.Thread(target=app.vision_loop)
    worker.start()
    done.wait(30)
    app.stop_event.set()
    worker.join(5)
    assert app.clips._events, "the play step's notes reach the clip buffer"
    song, _ = frames(cfg)
    for f, image in zip(song, _song_images(), strict=True):
        app.clips.add_frame(f.t, image)

    assert app._status(app.clock()).clip is None
    app.snap_clip()
    seen = app._status(app.clock()).clip
    assert seen is not None and seen.folder == "memories"
    app.snap_clip()  # a second press while saving does not start a second save
    assert app.clips.wait(120)
    assert app._message in (None, "still saving the last clip")
    final = app._status(app.clock()).clip
    assert final is not None and final.phase == "saved" and final.error is None
    assert (tmp_path / "memories" / final.name).stat().st_size > 0
    assert len(list((tmp_path / "memories").glob("*.mp4"))) == 1


def test_f8_before_any_frame_says_so(cfg):
    devices, _ = song_devices(cfg)
    app = LiveApp(cfg, Calibration(), devices)
    app.snap_clip()
    assert app._message == "nothing to save yet" and app._status(app.clock()).clip is None
