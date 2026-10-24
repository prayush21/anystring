"""The live app with fake devices: threads, recording, calibration, commands and CLI."""

from __future__ import annotations

import functools
import threading

import numpy as np
import pytest

from racketguitar.app import Devices, LiveApp, build_parser, gate_mode, main
from racketguitar.capture.camera import Frame
from racketguitar.config import Calibration, Config, load_calibration, load_config, save_config
from racketguitar.replay.harness import replay
from racketguitar.vision.markers import MarkerTracker
from tests.fakes import hand_obs, make_pose, place_hand, render_marker_scene, uv_to_px
from tests.fakes.hands import INDEX_TIP
from tests.integration.song import expected_note_count, frames


class FakeCamera:
    """Hands out prepared frames, then calls ``on_done``."""

    def __init__(self, items: list[Frame], on_done=lambda: None) -> None:
        self._items = list(items)
        self._on_done = on_done
        self.measured_fps = 60.0
        self.started = self.stopped = False

    def start(self):
        self.started = True
        return self

    def stop(self) -> None:
        self.stopped = True

    def read(self, timeout=None):
        if not self._items:
            self._on_done()
            return None
        return self._items.pop(0)


class FakeHands:
    def __init__(self, by_t: dict[float, tuple], fail: bool = False) -> None:
        self.by_t, self.fail, self.closed = by_t, fail, False

    def process(self, image, t):
        if self.fail:
            raise RuntimeError("landmarker crashed")
        return list(self.by_t.get(t, ()))

    def close(self) -> None:
        self.closed = True


class FakeAudio:
    def __init__(self) -> None:
        self.events = []
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def send(self, e) -> None:
        self.events.append(e)

    def detect_to_audio_ms(self):
        return np.full(len(self.events), 12.0)

    @property
    def last_detect_to_audio_ms(self):
        return 12.0 if self.events else None


@functools.cache
def _song_images() -> tuple[np.ndarray, ...]:
    song, _ = frames(Config())
    return tuple(render_marker_scene(**f.scene).image for f in song)


def song_devices(cfg: Config, on_done=lambda: None) -> tuple[Devices, FakeAudio]:
    song, _ = frames(cfg)
    images = _song_images()
    items = [Frame(img, f.t, i) for i, (img, f) in enumerate(zip(images, song, strict=True))]
    audio = FakeAudio()
    devices = Devices(
        camera=FakeCamera(items, on_done),
        markers=MarkerTracker(cfg.markers, cfg.filters),
        hands=FakeHands({f.t: f.hands for f in song}),
        audio=audio,
    )
    return devices, audio


def test_live_session_plays_records_and_replays(tmp_path, cfg):
    """The end-to-end check: rendered camera frames -> markers -> engine -> audio,
    recorded with lossless video, then replayed headlessly from data and from video."""
    cfg = cfg.with_overrides({"replay": {"video": True}})
    holder: dict[str, LiveApp] = {}
    devices, audio = song_devices(cfg, on_done=lambda: holder["app"].stop_event.set())
    app = LiveApp(cfg, Calibration(), devices, record_dir=tmp_path / "take")
    holder["app"] = app
    app.run(display=False, seconds=60)

    assert devices.camera.started and devices.camera.stopped and devices.hands.closed
    on = [e for e in audio.events if e.kind == "on"]
    assert len(on) == expected_note_count(cfg) == app.notes
    assert sum(e.kind == "damp" for e in audio.events) == 6

    first = replay(tmp_path / "take")
    assert first.matches, first.mismatches
    assert first.events == audio.events
    # Re-tracking from raw detections, and re-detecting on the lossless video (with
    # recorded detections standing in for frames the writer dropped), both reproduce it.
    for source in ("corners", "video"):
        again = replay(tmp_path / "take", markers_from=source)
        assert again.matches, (source, again.mismatches)


def test_vision_error_reaches_the_caller(cfg):
    devices, _ = song_devices(cfg)
    devices.hands = FakeHands({}, fail=True)
    app = LiveApp(cfg, Calibration(), devices)
    with pytest.raises(RuntimeError, match="landmarker crashed"):
        app.run(display=False, seconds=5)
    assert devices.camera.stopped


def _still_frames(cfg, shape: str, t0: float, n: int, fps: float = 30.0):
    """A still racket with the strumming hand held over the strings in ``shape``."""
    scene = render_marker_scene()
    out = []
    for k in range(n):
        t = t0 + k / fps
        pose = make_pose(t, origin=scene.pose.origin, angle_deg=-90.0, head_len=scene.pose.head_len)
        tip = uv_to_px(pose, 0.5, 0.45)  # parked beyond the high-E side
        hand = hand_obs(t, place_hand(shape, tip, anchor=INDEX_TIP, palm_px=80.0))
        out.append((Frame(scene.image, t, k), (hand,)))
    return out


def test_calibration_through_the_app(tmp_path, cfg):
    seq = _still_frames(cfg, "pick", 10.0, 2) + _still_frames(cfg, "pick", 10.1, 120)
    seq += _still_frames(cfg, "open", 14.2, 120)
    audio = FakeAudio()
    devices = Devices(
        camera=FakeCamera([]),
        markers=MarkerTracker(cfg.markers, cfg.filters),
        hands=FakeHands({f.t: h for f, h in seq}),
        audio=audio,
    )
    cal_path = tmp_path / "cal.toml"
    app = LiveApp(cfg, Calibration(), devices, calibrate=True, calibration_path=cal_path)
    views = [app.process(f) for f, _ in seq]
    assert views[0].calibration is not None
    assert app.flow is None, "calibration should have finished"
    cal = load_calibration(cal_path)
    assert cal.grip_closed_score is not None and cal.grip_open_score is not None
    assert cal.grip_closed_score < cal.grip_open_score
    assert app.pipeline.calibration == cal == app.calibration
    assert audio.events == []  # nothing plays while calibrating


def test_commands_apply_on_the_next_frame(tmp_path, cfg):
    cfg = cfg.with_overrides({"replay": {"record_dir": str(tmp_path / "rec"), "video": False}})
    devices, _ = song_devices(cfg)
    app = LiveApp(cfg, Calibration(), devices)
    frames_ = devices.camera._items
    app.command(app.toggle_recording)
    app.command(lambda: app.toggle_gate("grip"))
    app.command(lambda: app.toggle_gate("mic"))
    view = app.process(frames_.pop(0))
    assert gate_mode(app.cfg) == "free" and view.status.gate_mode == "free"
    assert view.status.recording
    assert "mic" in (view.status.message or "")
    app.process(frames_.pop(0))
    path = app.stop_recording()
    assert path is not None and (path / "session.json").exists()
    assert replay(path).matches


def test_gate_mode_names():
    c = Config()
    assert gate_mode(c) == "grip"
    assert gate_mode(c.with_overrides({"gate": {"mic": True}})) == "grip+mic"
    assert gate_mode(c.with_overrides({"gate": {"pick_grip": False, "mic": True}})) == "mic"
    assert gate_mode(c.with_overrides({"gate": {"pick_grip": False}})) == "free"


def test_cli_parses_defaults_and_subcommands():
    p = build_parser()
    a = p.parse_args([])
    assert a.command is None and a.mic is None and not a.calibrate
    assert p.parse_args(["--no-mic"]).mic is False
    assert p.parse_args(["play", "--mic", "--seconds", "2"]).seconds == 2.0
    assert p.parse_args(["replay", "x", "--from-mic"]).from_mic


def test_cli_replay(tmp_path, cfg, capsys):
    holder: dict[str, LiveApp] = {}
    devices, _ = song_devices(cfg, on_done=lambda: holder["app"].stop_event.set())
    holder["app"] = LiveApp(cfg, Calibration(), devices, record_dir=tmp_path / "take")
    holder["app"].run(display=False, seconds=60)
    with pytest.raises(SystemExit) as exit_:
        main(["replay", str(tmp_path / "take"), "--wav", str(tmp_path / "out.wav")])
    assert exit_.value.code == 0
    out = capsys.readouterr().out
    assert "matches the recording" in out and (tmp_path / "out.wav").exists()


def test_cli_play_without_model_explains(tmp_path, capsys):
    cfg = Config().with_overrides({"hands": {"model_path": str(tmp_path / "missing.task")}})
    save_config(cfg, tmp_path / "c.toml")
    with pytest.raises(SystemExit) as exit_:
        main(["--config", str(tmp_path / "c.toml"), "--no-display", "--no-audio"])
    assert exit_.value.code == 2
    assert "fetch-model" in capsys.readouterr().err


def test_cli_fetch_model_skips_existing(tmp_path, capsys):
    model = tmp_path / "hand.task"
    model.write_bytes(b"x")
    cfg = Config().with_overrides({"hands": {"model_path": str(model)}})
    save_config(cfg, tmp_path / "c.toml")
    with pytest.raises(SystemExit) as exit_:
        main(["fetch-model", "--config", str(tmp_path / "c.toml")])
    assert exit_.value.code == 0 and "already present" in capsys.readouterr().out


def test_vision_thread_publishes_views(cfg):
    done = threading.Event()
    devices, _ = song_devices(cfg, on_done=done.set)
    app = LiveApp(cfg, Calibration(), devices)
    t = threading.Thread(target=app.vision_loop)
    t.start()
    done.wait(30)
    app.stop_event.set()
    t.join(5)
    view = app.latest_view()
    assert view is not None and view.snapshot.visible and view.status.vision_ms is not None


def test_markerless_session_locks_tracks_and_plays(tmp_path, cfg):
    """No markers: lock with three clicks, follow the racket by optical flow, play the
    song, then replay the recording from its recorded poses."""
    from racketguitar.vision.flow import FlowTracker
    from tests.fakes.scenes import render_racket

    song, _ = frames(cfg)
    items = []
    for i, f in enumerate(song):
        u = np.asarray(f.pose.u)
        centre = np.asarray(f.pose.origin) + 0.5 * f.pose.head_len * u
        angle = float(np.degrees(np.arctan2(u[1], u[0])))
        scene = render_racket(center_px=tuple(centre), angle_deg=angle, head_len_px=f.pose.head_len)
        items.append(Frame(scene.image, f.t, i))
    first = song[0].pose
    tip = np.asarray(first.origin) + np.asarray(first.u) * first.head_len
    o, u = np.asarray(first.origin), np.asarray(first.u)
    grip = o - u * 0.15 * first.head_len  # where the song's chord sections start
    butt = o - u * first.head_len  # u = -1
    audio = FakeAudio()
    holder: dict[str, LiveApp] = {}
    devices = Devices(
        camera=FakeCamera(items, on_done=lambda: holder["app"].stop_event.set()),
        markers=MarkerTracker(cfg.markers, cfg.filters),
        hands=FakeHands({f.t: f.hands for f in song}),
        audio=audio,
        tracker=FlowTracker(cfg.markers, head_half_width=cfg.strings.head_half_width),
    )
    app = LiveApp(cfg, Calibration(), devices, record_dir=tmp_path / "take")
    holder["app"] = app
    app.request_lock(tip, first.origin, grip, butt)
    app.run(display=False, seconds=60)

    assert app.cfg.chords.handle_u_near == pytest.approx(-0.15, abs=0.01)
    assert app.cfg.chords.handle_u_far == pytest.approx(-1.0, abs=0.01)
    on = [e for e in audio.events if e.kind == "on"]
    assert len(on) == expected_note_count(cfg)
    assert replay(tmp_path / "take").matches
    with pytest.raises(ValueError, match="markerless"):
        replay(tmp_path / "take", markers_from="video")


class SlowHands(FakeHands):
    """A hand model whose detection is in flight when shutdown starts, and that refuses
    to close mid-detection (as MediaPipe's would, by blocking)."""

    def __init__(self) -> None:
        super().__init__({})
        self.busy = threading.Event()
        self.closed_while_busy = False

    def process(self, image, t):
        self.busy.set()
        threading.Event().wait(0.3)
        self.busy.clear()
        return []

    def close(self) -> None:
        self.closed_while_busy = self.busy.is_set()
        super().close()


def test_ctrl_c_shuts_down_cleanly(cfg):
    devices, _ = song_devices(cfg)
    hands = SlowHands()
    devices.hands = hands
    app = LiveApp(cfg, Calibration(), devices)

    def interrupt():
        hands.busy.wait(5)
        import _thread

        _thread.interrupt_main()  # what Ctrl-C does

    threading.Thread(target=interrupt, daemon=True).start()
    app.run(display=False, seconds=30)  # returns instead of raising KeyboardInterrupt
    assert hands.closed and not hands.closed_while_busy
    assert devices.camera.stopped


def test_recording_follows_setting_changes(tmp_path, cfg):
    """Changing settings mid-recording must never leave a recording whose saved settings
    differ from what it played with: before any frame they are updated, after frames
    the recording is split so each part replays exactly."""
    devices, _ = song_devices(cfg)
    app = LiveApp(cfg, Calibration(), devices, record_dir=tmp_path / "take")
    items = devices.camera._items
    app.toggle_gate("grip")  # before any frame: the recording just takes the new settings
    for _ in range(40):
        app.process(items.pop(0))
    app.toggle_gate("grip")  # after frames: split
    for _ in range(40):
        app.process(items.pop(0))
    app.stop_recording()
    first, second = tmp_path / "take", tmp_path / "take-part2"
    assert first.exists() and second.exists()
    a, b = replay(first), replay(second)
    assert a.matches and b.matches
    assert gate_mode(load_config(first / "config.toml")) == "free"
    assert gate_mode(load_config(second / "config.toml")) == "grip"
