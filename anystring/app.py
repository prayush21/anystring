"""App wiring and CLI: ``uv run anystring``.

Threads (PID "Solution design"): the camera keeps the newest frame on its own thread;
the vision thread here runs hands, racket tracking, gate and engine and sends notes to
the audio callback; the mic runs in its PortAudio callback; the main thread only draws the
overlay, because macOS requires GUI work there.

The racket is found by clicking it (top of the head, throat, grip start, handle end) and then
followed by optical flow; there are no markers.

Keys in the play window: q / Esc quit, r start/stop recording, c calibrate,
g toggle the pick-grip gate, m toggle the mic gate (needs --mic), l re-lock,
d debug view (with the live grip meter), h help. [ and ] lower / raise the grip's off
threshold, , and . the arm threshold, for trying thresholds while playing; they are
shares of the calibrated pick-grip -> open-hand span, and a recording splits into a new
part at each change so every part replays with its own thresholds.
Song mode: s talk ("let's play take on me", "start", "stop"), SPACE start/stop the
song, x leave the song.
Clips: F8 (or f, since a MacBook's F8 is a media key unless Fn is held) saves the last
``[clip] seconds`` of the window and the guitar's sound to the ``memories/`` folder; the card
in the corner shows it saving, then saved. The save runs on a worker thread.

Song mode reads chord sheets from ``~/.anystring/songs``. A ``SongPlayer`` decides
which chords label the handle sections over the song; each relabel goes through
``FrameInput.chords`` so it is part of the play step and of any recording. Voice
requests (recording and speech-to-text) run on a worker thread, never on the play path.
"""

from __future__ import annotations

import argparse
import contextlib
import queue
import shutil
import ssl
import sys
import threading
import time
import urllib.request
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import certifi
import numpy as np

from anystring.audio.output import AudioOutput
from anystring.audio.probe import ProbeTone
from anystring.capture.camera import Camera, CameraError, Frame
from anystring.capture.mic import MicStream
from anystring.config import (
    Calibration,
    Config,
    home_dir,
    load_calibration,
    load_config,
    save_calibration,
)
from anystring.pipeline import FrameInput, Pipeline
from anystring.play.gate import grip_calibrated
from anystring.replay.clip import ClipBuffer
from anystring.replay.demo import DemoError, DemoRecorder
from anystring.replay.harness import render_session, replay
from anystring.replay.record import Recorder, TeeDetector
from anystring.song.library import find_song, songs_dir
from anystring.song.player import SongPlayer
from anystring.song.sheet import Song
from anystring.song.voice import VoiceError, parse_command, record_clip, transcribe, wav_bytes
from anystring.types import HandObs, NoteEvent, RacketPose
from anystring.ui.calibrate import CalibrationFlow
from anystring.ui.clip_hud import ClipStatus, draw_clip_hud
from anystring.ui.overlay import CalibrationView, Overlay, Status, View
from anystring.vision.flow import FlowTracker
from anystring.vision.hands import MODEL_URL, HandModelMissingError, HandTracker

__all__ = ["Devices", "LiveApp", "main", "gate_mode", "model_target", "clip_target", "is_snap_key"]

WINDOW = "Anystring"
HELP = (
    "q quit  r record  c calibrate  g grip gate  m mic gate  l re-lock  d debug  h help"
    "  [ ] grip off  , . grip arm  s talk  SPACE song start/stop  x leave song  F8 snap clip"
)
GRIP_STEP = 0.05  # [ ] , . move a grip threshold by this share of the closed -> open span
LOCK_PROMPTS = (
    "Click the TOP of the racket head",
    "Click the THROAT, where the head meets the shaft",
    "Click where the GRIP starts (top of the handle wrap)",
    "Click the END of the handle (the very bottom)",
)
MESSAGE_S = 3.0  # how long status messages stay on screen
# What ``cv2.waitKeyEx`` returns for F8: macOS (Cocoa's private-use code), X11/GTK, Windows.
# ``f`` also snaps, because a MacBook's F8 is a media key unless Fn is held.
SNAP_KEYS = frozenset({0xF70B, 0xFFC5, 0x770000, ord("f")})
_REPO_ROOT = Path(__file__).resolve().parents[1]


class CameraLike(Protocol):
    measured_fps: float

    def start(self) -> Any: ...
    def stop(self) -> None: ...
    def read(self, timeout: float | None = None) -> Frame | None: ...


class HandsLike(Protocol):
    def process(self, image: np.ndarray, t: float) -> list[HandObs]: ...
    def close(self) -> None: ...


@dataclass
class Devices:
    """What the live app drives. Tests pass fakes; ``open_devices`` builds real ones."""

    camera: CameraLike
    hands: HandsLike
    tracker: FlowTracker  # follows the racket after it is clicked
    audio: AudioOutput | None = None
    mic: MicStream | None = None
    tee: TeeDetector | None = None


def gate_mode(cfg: Config) -> str:
    g = cfg.gate
    return {(True, True): "grip+mic", (True, False): "grip", (False, True): "mic"}.get(
        (g.pick_grip, g.mic), "free"
    )


class LiveApp:
    """One play session. ``process`` is the vision thread's per-frame work."""

    def __init__(
        self,
        cfg: Config,
        calibration: Calibration,
        devices: Devices,
        *,
        calibrate: bool = False,
        calibration_path: Path | None = None,
        record_dir: Path | None = None,
        demo_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        song: Song | None = None,
        song_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.calibration = calibration
        self.dev = devices
        self.clock = clock
        self.calibration_path = calibration_path
        self.pipeline = Pipeline(cfg, calibration)
        self.flow = CalibrationFlow(cfg, calibration, devices.tee) if calibrate else None
        self._pending_lock: tuple[np.ndarray, ...] | None = None
        self.recorder: Recorder | None = None
        self.demo = DemoRecorder(demo_path, cfg) if demo_path is not None else None
        self.clips = ClipBuffer(cfg, clock)
        self.notes = 0
        self.error: BaseException | None = None  # set if the vision thread died
        self.stop_event = threading.Event()
        self._commands: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._view_lock = threading.Lock()
        self._view: View | None = None
        self._onsets: deque[float] = deque(maxlen=16)
        self._message: str | None = None
        self._message_until = 0.0
        self._vision_ms: float | None = None
        self.song_dir = song_dir
        self.player: SongPlayer | None = None
        self._song_sent: tuple | None = None  # chord set last sent to the play step
        self._song_start = False  # start the song on the next frame
        self._frame_t = 0.0  # capture time of the latest frame (song time base)
        self._listening = threading.Event()
        # Push-to-talk: record a clip and return what was said. Tests replace it.
        self.hear: Callable[[], str] = self._hear
        if song is not None:
            self._load_song(song)
        if record_dir is not None:
            self._start_recording(record_dir)

    # -- vision thread ------------------------------------------------------------------

    def process(self, frame: Frame) -> View:
        t0 = self.clock()
        while True:
            try:
                self._commands.get_nowait()()
            except queue.Empty:
                break
        hands = tuple(self.dev.hands.process(frame.image, frame.t))
        pose = self._track(frame, hands)
        onsets = tuple(self.dev.mic.drain()) if self.dev.mic is not None else ()
        self._onsets.extend(o.t for o in onsets)
        self._frame_t = frame.t
        song_state, chords = self._song_step(frame.t)
        inp = FrameInput(frame.t, t0, pose, hands, onsets, chords)
        events = self.pipeline.step(inp)
        snap = self.pipeline.snapshot()

        cal_view = None
        if self.flow is not None:
            self.flow.update(frame.t, pose, snap.strumming)
            cal_view = CalibrationView(
                self.flow.instruction, self.flow.progress, self.flow.error, self.flow.warning
            )
            if self.flow.done:
                self._finish_calibration()
        else:
            self._play(events)
            if self.demo is not None:
                self.demo.add_events(events)
            self.clips.add_events(events)
            if self.recorder is not None:
                image = frame.image if self.cfg.replay.video else None
                self.recorder.add(inp, events, image)

        self._vision_ms = (self.clock() - t0) * 1000.0
        view = View(
            frame.image,
            frame.t,
            pose,
            snap,
            self._status(t0),
            cal_view,
            track_points=self.dev.tracker.points.copy(),
            song=self.player,
            song_state=song_state,
        )
        with self._view_lock:
            self._view = view
        return view

    def vision_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                frame = self.dev.camera.read(timeout=0.25)
                if frame is not None:
                    self.process(frame)
        except Exception as exc:  # surface to the main thread, then stop
            self._set_message(f"vision thread stopped: {exc}")
            self.error = exc
            self.stop_event.set()

    def latest_view(self) -> View | None:
        with self._view_lock:
            return self._view

    # -- commands (any thread; applied at the start of the next frame) ------------------

    def command(self, fn: Callable[[], None]) -> None:
        self._commands.put(fn)

    def toggle_recording(self) -> None:
        if self.recorder is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self._start_recording(Path(self.cfg.replay.record_dir) / stamp)
        else:
            path = self.stop_recording()
            self._set_message(f"saved {path}")

    def snap_clip(self) -> None:
        """F8: save the last ``clip.seconds`` to the clip folder. Main thread; the encode
        runs on its own thread and the card in the window shows how it is going."""
        if self.clips.saving:
            self._set_message("still saving the last clip")
            return
        try:
            save = self.clips.snap(clip_target(self.cfg))
        except DemoError as exc:
            self._set_message(str(exc))
            return
        if save is None:
            self._set_message("nothing to save yet")

    def start_calibration(self) -> None:
        self.flow = CalibrationFlow(self.cfg, self.calibration, self.dev.tee)

    def request_lock(self, tip, throat, grip, handle_end) -> None:
        """Lock onto the racket at these clicked frame pixels (any
        thread; applied to the next frame). The chord sections span grip -> handle end."""
        pts = tuple(np.asarray(p, dtype=float) for p in (tip, throat, grip, handle_end))
        self.command(lambda: setattr(self, "_pending_lock", pts))

    def toggle_gate(self, which: str) -> None:
        if which == "mic" and self.dev.mic is None:
            self._set_message("mic gate needs the mic: restart with --mic")
            return
        key = "mic" if which == "mic" else "pick_grip"
        cfg = self.cfg.with_overrides({"gate": {key: not getattr(self.cfg.gate, key)}})
        self._rebuild(cfg, self.calibration)
        self._set_message(f"gate: {gate_mode(cfg)}")

    def nudge_grip(self, which: str, step: float) -> None:
        """Move the grip's arm (``"arm"``) or off (``"off"``) threshold by ``step``, as a
        share of the calibrated closed -> open span. Arming always stays below off."""
        g = self.cfg.gate
        arm, off = g.grip_arm_below, g.grip_release_above
        if which == "arm":
            arm = min(round(arm + step, 2), round(off - GRIP_STEP, 2))
        else:
            off = max(round(off + step, 2), round(arm + GRIP_STEP, 2))
        cfg = self.cfg.with_overrides({"gate": {"grip_arm_below": arm, "grip_release_above": off}})
        self._rebuild(cfg, self.calibration)
        self._set_message(f"grip: arm below {arm:.0%}, off above {off:.0%} (d shows the meter)")

    def toggle_song(self) -> None:
        """SPACE: start the loaded song (or play it again), or stop it."""
        p = self.player
        if p is None:
            self._set_message('no song loaded: press s and say "let\'s play ..."')
        elif p.started and not self._song_done():
            p.stop()
            self._set_message("song stopped: SPACE to start again")
        else:
            self._song_start = True

    def leave_song(self) -> None:
        if self.player is not None:
            self.player = None
            self._set_message("song mode off: back to the default chords")

    def listen(self) -> None:
        """Push-to-talk: record and transcribe on a worker thread, then act on it."""
        if self._listening.is_set():
            return
        self._listening.set()
        self._set_message(f"listening for {self.cfg.song.listen_s:g} s ...")
        threading.Thread(target=self._listen_worker, name="voice", daemon=True).start()

    def handle_heard(self, text: str) -> None:
        """Act on a transcribed request (any thread; session changes go via commands)."""
        cmd = parse_command(text)
        if cmd.kind == "start":
            self.command(lambda: self._start_if_idle())
        elif cmd.kind == "stop":
            self.command(lambda: self.player and self.player.stop())
        elif cmd.kind == "restart":
            self.command(self._restart_song)
        else:
            query = cmd.query or cmd.text
            match = find_song(query, self.song_dir, self.cfg.song.match_min_score)
            if match is None:
                self.command(lambda: self._set_message(f'heard "{text}": no song file matches'))
            else:
                self.command(lambda: self._load_song(match.song))

    def stop_recording(self) -> Path | None:
        rec, self.recorder = self.recorder, None
        if rec is None:
            return None
        if self.dev.audio is not None:
            lat = self.dev.audio.detect_to_audio_ms()
            if lat.size:
                rec.session.meta["detect_to_audio_ms"] = {
                    "p50": float(np.percentile(lat, 50)),
                    "p95": float(np.percentile(lat, 95)),
                    "n": int(lat.size),
                }
        return rec.stop()

    # -- main thread --------------------------------------------------------------------

    def run(self, *, display: bool = True, seconds: float | None = None) -> None:
        """Start devices and the vision thread, then draw (or just wait) until done."""
        started: list[Any] = []
        vision: threading.Thread | None = None
        try:
            for dev in (self.dev.audio, self.dev.mic, self.dev.camera):
                if dev is not None:
                    dev.start()
                    started.append(dev)
            vision = threading.Thread(target=self.vision_loop, name="vision", daemon=True)
            vision.start()
            deadline = None if seconds is None else self.clock() + seconds
            if display:
                self._display_loop(deadline)
            else:
                while not self.stop_event.is_set():
                    if deadline is not None and self.clock() >= deadline:
                        break
                    self.stop_event.wait(0.05)
        except KeyboardInterrupt:  # Ctrl-C quits like q
            print("\nstopping... (releasing the camera can take a second)", flush=True)
        finally:
            # Stop the vision thread before anything it uses: closing the hand model
            # while a detection is in flight deadlocks inside MediaPipe.
            self.stop_event.set()
            if vision is not None:
                vision.join(timeout=3.0)
            vision_done = vision is None or not vision.is_alive()
            if self.clips.saving:
                print("finishing the clip ...", flush=True)
                self.clips.wait(120.0)
            if vision_done:
                path = self.stop_recording()
                if path is not None:
                    print(f"recording saved to {path}")
                demo, self.demo = self.demo, None
                if demo is not None:
                    print("encoding the demo video ...", flush=True)
                    saved = demo.stop()
                    print(f"demo saved to {saved}" if saved else "demo: nothing was captured")
            for dev in reversed(started):
                with contextlib.suppress(Exception):
                    dev.stop()
            if vision_done:
                with contextlib.suppress(Exception):
                    self.dev.hands.close()
        if self.error is not None:
            raise self.error

    def _display_loop(self, deadline: float | None) -> None:
        import cv2

        overlay = Overlay(self.cfg)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        period = 1.0 / max(1, self.cfg.ui.max_fps)
        last = None
        clicks: list[tuple[float, float]] = []
        locking = [True]  # the session starts by asking for the clicks
        debug = [False]  # tracked points and grip score

        def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
            if event != cv2.EVENT_LBUTTONDOWN or not locking[0] or last is None:
                return
            w = last.image.shape[1]
            clicks.append((w - 1 - x if self.cfg.ui.mirror else x, y))
            if len(clicks) == len(LOCK_PROMPTS):
                self.request_lock(*clicks)
                clicks.clear()
                locking[0] = False

        def relock() -> None:
            clicks.clear()
            locking[0] = True

        cv2.setMouseCallback(WINDOW, on_mouse)
        keys: dict[int, Callable[[], None]] = {
            ord("r"): self.toggle_recording,
            ord("c"): self.start_calibration,
            ord("g"): lambda: self.toggle_gate("grip"),
            ord("m"): lambda: self.toggle_gate("mic"),
            ord("h"): lambda: self._set_message(HELP),
            ord("["): lambda: self.nudge_grip("off", -GRIP_STEP),
            ord("]"): lambda: self.nudge_grip("off", GRIP_STEP),
            ord(","): lambda: self.nudge_grip("arm", -GRIP_STEP),
            ord("."): lambda: self.nudge_grip("arm", GRIP_STEP),
            ord(" "): self.toggle_song,
            ord("x"): self.leave_song,
        }
        try:
            while not self.stop_event.is_set():
                if deadline is not None and self.clock() >= deadline:
                    break
                tick = self.clock()
                view = self.latest_view()
                if view is not None and view is not last:
                    prompt = LOCK_PROMPTS[len(clicks)] if locking[0] else None
                    img = overlay.draw(view, clicks=tuple(clicks), prompt=prompt, debug=debug[0])
                    if self.demo is not None and not locking[0]:  # not the lock clicks
                        self.demo.add_frame(view.t, img)
                    self.clips.add_frame(view.t, img)
                    draw_clip_hud(img, view.status.clip, view.t)  # after: kept out of videos
                    cv2.imshow(WINDOW, img)
                    last = view
                raw = cv2.waitKeyEx(1)
                key = raw if 0 <= raw < 256 else -1  # function keys come back as big codes
                if is_snap_key(raw):
                    self.snap_clip()
                if key in (ord("q"), 27):
                    break
                if key == ord("l"):
                    relock()
                if key == ord("d"):
                    debug[0] = not debug[0]
                if key == ord("s"):
                    self.listen()
                if key in keys:
                    self.command(keys[key])
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                time.sleep(max(0.0, period - (self.clock() - tick)))
        finally:
            cv2.destroyAllWindows()

    # -- helpers ------------------------------------------------------------------------

    def _song_step(self, t: float):
        """This frame's song state, and the chord set to send if it changed."""
        p = self.player
        if p is None:
            if self._song_sent is None:
                return None, None
            self._song_sent = None  # song left: back to the configured chords
            return None, self.cfg.chords.chords
        if self._song_start:
            self._song_start = False
            p.start(t)
        state = p.state(t)
        if state.chords == self._song_sent:
            return state, None
        self._song_sent = state.chords
        return state, state.chords

    def _song_done(self) -> bool:
        p = self.player
        return p is not None and p.state(self._frame_t).phase == "done"

    def _start_if_idle(self) -> None:
        if self.player is None:
            self._set_message("no song loaded yet")
        elif not self.player.started or self._song_done():
            self._song_start = True

    def _restart_song(self) -> None:
        if self.player is not None:
            self.player.stop()
            self._song_start = True

    def _load_song(self, song: Song) -> None:
        try:
            player = SongPlayer(song, self.cfg)
        except ValueError as exc:  # an unknown chord or an empty sheet
            self._set_message(f"{song.title}: {exc}")
            return
        self.player = player
        self._song_sent = None
        by = f" by {song.artist}" if song.artist else ""
        notes = f" ({'; '.join(song.notes)})" if song.notes else ""
        self._set_message(f"{song.title}{by} loaded{notes}")

    def _hear(self) -> str:
        clip = record_clip(self.cfg.song.listen_s, self.cfg.mic.device)
        return transcribe(wav_bytes(clip), self.cfg.song)

    def _listen_worker(self) -> None:
        try:
            text = self.hear()
        except (VoiceError, OSError) as exc:
            msg = str(exc)
            self.command(lambda: self._set_message(msg))
        else:
            self.command(lambda: self._set_message(f'heard "{text}"'))
            self.handle_heard(text)
        finally:
            self._listening.clear()

    def _track(self, frame: Frame, hands: tuple[HandObs, ...]) -> RacketPose:
        tracker = self.dev.tracker
        exclude = [h.raw for h in hands]
        if self._pending_lock is not None:
            tip, throat, grip, handle_end = self._pending_lock
            self._pending_lock = None
            pose = tracker.lock(frame.image, tip, throat, frame.t, exclude, handle_end=handle_end)
            if pose.visible:
                # The grip and handle-end clicks place the chord sections on the grip only.
                near, far = (
                    float(np.dot(p - throat, pose.u)) / pose.head_len for p in (grip, handle_end)
                )
                if far < near - 0.1 and near < 0.0:
                    chords = {"handle_u_near": near, "handle_u_far": far}
                    self._rebuild(self.cfg.with_overrides({"chords": chords}), self.calibration)
                    self._set_message(f"locked on; chords on the grip, u {near:.2f} to {far:.2f}")
                else:
                    self._set_message("locked on; grip clicks looked wrong, chords unchanged")
            return pose
        return tracker.update(frame.image, frame.t, exclude)

    def _play(self, events: list[NoteEvent]) -> None:
        if self.dev.audio is None:
            return
        for e in events:
            self.dev.audio.send(e)
        self.notes += sum(e.kind == "on" for e in events)

    def _start_recording(self, path: Path) -> None:
        self.recorder = Recorder(
            path, self.cfg, self.calibration, video=self.cfg.replay.video, mic=self.dev.tee
        )
        self.recorder.session.meta["pose_source"] = "flow"
        self._song_sent = None  # the recording's first frame carries the song's chords
        self._set_message(f"recording to {path}")

    def _finish_calibration(self) -> None:
        assert self.flow is not None and self.flow.result is not None
        cal = self.flow.result
        if self.dev.audio is not None:
            lat = self.dev.audio.detect_to_audio_ms()
            if lat.size >= 5:
                cal = replace(cal, detect_to_audio_ms=float(np.median(lat)))
        warning = self.flow.warning
        self.flow = None
        save_calibration(cal, self.calibration_path)
        self._rebuild(self.cfg, cal)
        self._set_message(warning or "calibration saved")

    def _rebuild(self, cfg: Config, calibration: Calibration) -> None:
        """Apply new settings. A recording follows them so it always replays exactly: if
        it has no frames yet its settings are updated, otherwise it is saved and a new
        part starts in ``<dir>-2``, ``<dir>-3``, ..."""
        self.cfg, self.calibration = cfg, calibration
        self.pipeline = Pipeline(cfg, calibration)
        self._song_sent = None  # a new pipeline starts on the configured chords
        if self.player is not None:
            self.player = self._replan(self.player)
        rec = self.recorder
        if rec is None:
            return
        if not rec.session.frames:
            rec.session.config, rec.session.calibration = cfg, calibration
            return
        done = self.stop_recording()
        base = rec.path.name.rsplit("-part", 1)[0]
        n = 2
        while (rec.path.parent / f"{base}-part{n}").exists():
            n += 1
        self._start_recording(rec.path.parent / f"{base}-part{n}")
        self._set_message(f"settings changed: saved {done}, recording continues")

    def _replan(self, player: SongPlayer) -> SongPlayer:
        """The same song under new settings, keeping its start time."""
        fresh = SongPlayer(player.song, self.cfg)
        if player.start_time is not None:
            fresh.start(player.start_time)
        return fresh

    def _status(self, now: float) -> Status:
        audio, mic = self.dev.audio, self.dev.mic
        msg = self._message if now < self._message_until else None
        g, clf = self.cfg.gate, self.pipeline.gate.classifier
        lo, hi = clf.close_below, clf.open_above
        span = (hi - lo) / max(g.grip_release_above - g.grip_arm_below, 1e-9)
        return Status(
            fps=float(getattr(self.dev.camera, "measured_fps", 0.0)),
            vision_ms=self._vision_ms,
            latency_ms=audio.last_detect_to_audio_ms if audio is not None else None,
            gate_mode=gate_mode(self.cfg),
            mic_level=mic.level if mic is not None else None,
            onsets=tuple(self._onsets),
            recording=self.recorder is not None,
            message=msg,
            clip=self._clip_status(),
            grip_score=self.pipeline.gate.grip_score,
            grip_band=(lo, hi),
            grip_span=(lo - g.grip_arm_below * span, lo + (1 - g.grip_arm_below) * span),
        )

    def _clip_status(self) -> ClipStatus | None:
        save = self.clips.current
        if save is None:
            return None
        finished = save.finished  # read first: the worker sets the error before it
        phase = "saving" if finished is None else "failed" if save.error else "saved"
        return ClipStatus(
            phase,
            save.started,
            finished,
            save.progress,
            save.seconds,
            save.path.parent.name,
            save.path.name,
            save.error,
        )

    def _set_message(self, text: str) -> None:
        self._message, self._message_until = text, self.clock() + MESSAGE_S


# -- CLI ------------------------------------------------------------------------------


def model_target(cfg: Config) -> Path:
    p = Path(cfg.hands.model_path)
    return p if p.is_absolute() else _REPO_ROOT / p


def clip_target(cfg: Config) -> Path:
    p = Path(cfg.clip.folder).expanduser()
    return p if p.is_absolute() else _REPO_ROOT / p


def is_snap_key(raw: int) -> bool:
    """Whether a ``cv2.waitKeyEx`` code is the clip key."""
    return raw in SNAP_KEYS


def open_devices(cfg: Config, *, audio: bool, mic: bool, probe_tone: bool = False) -> Devices:
    tee = TeeDetector(cfg.mic) if mic else None
    synth = ProbeTone.from_config(cfg) if probe_tone else None
    return Devices(
        camera=Camera(cfg.camera),
        hands=HandTracker(cfg.hands, cfg.filters),
        audio=AudioOutput.from_config(cfg, synth) if audio else None,
        mic=MicStream(cfg.mic, detector=tee) if mic else None,
        tee=tee,
        tracker=FlowTracker(cfg.tracking, head_half_width=cfg.strings.head_half_width),
    )


def _cmd_play(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else load_config()
    if args.mic is not None:
        cfg = cfg.with_overrides({"gate": {"mic": args.mic}})
    if args.speed is not None:
        cfg = cfg.with_overrides({"song": {"speed": args.speed}})
    cal_path = Path(args.calibration) if args.calibration else home_dir() / "calibration.toml"
    cal = load_calibration(cal_path)
    calibrate = args.calibrate or (
        not grip_calibrated(cal) and not args.skip_calibration and cfg.gate.pick_grip
    )
    try:
        devices = open_devices(
            cfg,
            audio=not args.no_audio,
            mic=cfg.gate.mic,
            probe_tone=args.probe_tone,
        )
    except HandModelMissingError as exc:
        print(f"{exc}\nRun: uv run anystring fetch-model", file=sys.stderr)
        return 2
    song = None
    if args.song:
        match = find_song(args.song, None, cfg.song.match_min_score)
        if match is None:
            print(f"no song sheet matches {args.song!r} in {songs_dir()}", file=sys.stderr)
            return 2
        song = match.song
    try:
        app = LiveApp(
            cfg,
            cal,
            devices,
            calibrate=calibrate,
            calibration_path=cal_path,
            record_dir=Path(args.record) if args.record else None,
            demo_path=Path(args.demo) if args.demo else None,
            song=song,
        )
        app.run(display=not args.no_display, seconds=args.seconds)
    except CameraError as exc:
        print(f"camera: {exc}", file=sys.stderr)
        return 3
    except DemoError as exc:
        print(f"demo: {exc}", file=sys.stderr)
        return 4
    if devices.audio is not None:
        lat = devices.audio.detect_to_audio_ms()
        if lat.size:
            p50, p95 = np.percentile(lat, [50, 95])
            print(f"{app.notes} notes, detect-to-audio p50 {p50:.1f} ms, p95 {p95:.1f} ms")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else None
    result = replay(
        args.session,
        config=cfg,
        onsets_from="mic" if args.from_mic else "recorded",
    )
    print(f"{len(result.frames)} frames, {len(result.events)} events")
    for line in result.mismatches:
        print(line)
    print("matches the recording" if result.matches else "differs from the recording")
    if args.wav:
        from scipy.io import wavfile

        from anystring.replay.record import load_session

        session_cfg = cfg or load_session(args.session).config
        audio = render_session(result.events, session_cfg)
        wavfile.write(args.wav, session_cfg.audio.sample_rate, audio)
        print(f"wrote {args.wav}")
    return 0 if result.matches or cfg is not None else 1


def _cmd_song(args: argparse.Namespace) -> int:
    """Print how a song sheet will play: timing, chords per line and handle labels."""
    cfg = load_config(args.config) if args.config else load_config()
    if args.speed is not None:
        cfg = cfg.with_overrides({"song": {"speed": args.speed}})
    match = find_song(args.query, None, cfg.song.match_min_score)
    if match is None:
        print(f"no song sheet matches {args.query!r} in {songs_dir()}", file=sys.stderr)
        return 2
    song = match.song
    p = SongPlayer(song, cfg)
    spb = p.seconds_per_beat
    print(f"{song.title} by {song.artist or '?'}  ({song.path})")
    print(
        f"{song.tempo:g} BPM x {p.speed:.0%} = {p.bpm:.0f} BPM, {song.beats_per_bar}/4, "
        f"capo {song.capo}, {p.timeline.total_beats * spb:.0f} s"
    )
    for note in song.notes:
        print(f"  note: {note}")
    section = None
    for i, line in enumerate(p.timeline.lines):
        if line.section != section:
            section = line.section
            print(f"[{section or '-'}]")
        chords = " ".join(f"{c.name}({c.beats:g})" for c in p.timeline.chords if c.line == i)
        text = "" if args.no_lyrics else f"  {line.text}"
        print(f"  {line.beat * spb:6.1f}s  {chords or '(holds)':28s}{text}")
    print("handle sections, head -> handle end:")
    for change in p.plan:
        when = "start" if change.beat == float("-inf") else f"{change.beat * spb:.1f}s"
        print(f"  {when:>7}  {'  '.join(change.slots)}")
    for name, notes in p.voicings.items():
        print(f"  {name:6s} {notes}")
    return 0


def _cmd_fetch_model(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else load_config()
    target = model_target(cfg)
    if target.exists() and not args.force:
        print(f"already present: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    print(f"downloading {MODEL_URL}\n  -> {target}")
    # python.org's macOS Python has no CA certificates until its "Install Certificates"
    # script runs, so verify against certifi's bundle instead of the system default.
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(MODEL_URL, context=ctx, timeout=60) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(target)
    print(f"done ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anystring", description="Play a tennis racket.")
    sub = p.add_subparsers(dest="command")

    play = sub.add_parser("play", help="play (the default)")
    _play_args(play)
    _play_args(p)

    cal = sub.add_parser("calibrate", help="run calibration, then play")
    _play_args(cal)

    rp = sub.add_parser("replay", help="replay a recorded session headlessly")
    rp.add_argument("session", help="session directory")
    rp.add_argument("--config", help="settings to try instead of the recorded ones")
    rp.add_argument("--from-mic", action="store_true", help="re-detect onsets from mic.wav")
    rp.add_argument("--wav", help="render the replayed notes to this WAV")

    sg = sub.add_parser("song", help="show how a song sheet will play")
    sg.add_argument("query", help="song title or sheet file")
    sg.add_argument("--config")
    sg.add_argument("--speed", type=float, help="fraction of the sheet's tempo")
    sg.add_argument("--no-lyrics", action="store_true", help="chords and timing only")

    fm = sub.add_parser("fetch-model", help="download the MediaPipe hand model")
    fm.add_argument("--config")
    fm.add_argument("--force", action="store_true")
    return p


def _play_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="settings TOML (default ~/.anystring/config.toml)")
    p.add_argument("--calibration", help="calibration TOML to read and write")
    p.add_argument("--calibrate", action="store_true", help="run calibration first")
    p.add_argument("--skip-calibration", action="store_true", help="play uncalibrated")
    p.add_argument("--record", metavar="DIR", help="record the session to DIR")
    p.add_argument(
        "--demo", metavar="FILE.mp4", help="save the window and the guitar sound as a video"
    )
    p.add_argument("--mic", dest="mic", action="store_true", default=None, help="mic gate on")
    p.add_argument("--no-mic", dest="mic", action="store_false", help="mic gate off")
    p.add_argument("--no-audio", action="store_true", help="no sound output")
    p.add_argument(
        "--probe-tone", action="store_true", help="beep instead of guitar (tools/latency_probe)"
    )
    p.add_argument("--no-display", action="store_true", help="no window (for smoke tests)")
    p.add_argument("--seconds", type=float, help="stop after this many seconds")
    p.add_argument("--song", help="load a song sheet (title or file) in song mode")
    p.add_argument("--speed", type=float, help="song speed, fraction of its tempo")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command = args.command or "play"
    if command == "calibrate":
        args.calibrate = True
    handler = {
        "play": _cmd_play,
        "calibrate": _cmd_play,
        "replay": _cmd_replay,
        "song": _cmd_song,
        "fetch-model": _cmd_fetch_model,
    }[command]
    try:
        code = handler(args)
    except KeyboardInterrupt:  # a second Ctrl-C during shutdown: just leave
        code = 130
    raise SystemExit(code)
