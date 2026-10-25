"""Demo capture: the overlay window and the guitar's sound in one MP4.

The window is piped to ffmpeg as constant-frame-rate video, timed by the camera's
capture clock. The sound is not recorded from a speaker: the notes the play step emitted
are rendered offline by the same synth the live app uses (as ``replay --wav`` does), at
their exact times, and muxed with the video when capture stops. So picture and sound are
in sync without a loopback audio device, and the take is free of room noise.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from anystring.audio.output import render_offline
from anystring.audio.synth import make_synth
from anystring.config import Config
from anystring.types import NoteEvent

__all__ = ["DemoError", "DemoRecorder", "ffmpeg_path"]

FPS = 30


class DemoError(RuntimeError):
    pass


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise DemoError("saving video needs ffmpeg on the PATH (macOS: brew install ffmpeg)")
    return path


class DemoRecorder:
    """Collect window frames and note events; ``stop`` writes ``path`` (an .mp4).

    If the encoder falls behind, frames are dropped (counted in ``dropped``) rather than
    stalling the window; the timeline repeats the last frame to cover them. Pass
    ``drop_frames=False`` to wait instead (offline or test capture).
    """

    def __init__(
        self, path: str | Path, cfg: Config, *, fps: int = FPS, drop_frames: bool = True
    ) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self.fps = fps
        self._drop = drop_frames
        self._ffmpeg = ffmpeg_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._video_path = self.path.with_name(self.path.stem + ".video.mp4")
        self._events: list[NoteEvent] = []
        self._q: queue.Queue[tuple[float, np.ndarray] | None] = queue.Queue(maxsize=8)
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._t0: float | None = None
        self._written = 0  # frames sent to ffmpeg
        self.dropped = 0  # frames the writer could not keep up with
        self.error: str | None = None

    # -- called by the app --------------------------------------------------------------

    def add_events(self, events: list[NoteEvent]) -> None:
        self._events.extend(events)

    def add_frame(self, t: float, image: np.ndarray) -> None:
        """One window image taken at capture time ``t`` (main thread)."""
        if self._thread is None:
            self._start(image.shape[1], image.shape[0])
        try:
            self._q.put((t, image.copy()), block=not self._drop)
        except queue.Full:
            self.dropped += 1

    def stop(self) -> Path | None:
        """Finish the video, render the audio and mux them. None if nothing was captured."""
        if self._thread is None or self._proc is None:
            return None
        self._q.put(None)
        self._thread.join()
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0 or self._t0 is None:
            raise DemoError(f"ffmpeg failed while encoding video: {self.error or 'see above'}")
        duration = self._written / self.fps
        wav = self.path.with_name(self.path.stem + ".audio.wav")
        sr = self.cfg.audio.sample_rate
        audio = render_offline(
            make_synth(self.cfg),
            self._events,
            duration,
            start_time=self._t0,
            master_gain=self.cfg.audio.master_gain,
            clip_knee=self.cfg.audio.clip_knee,
        )
        wavfile.write(str(wav), sr, audio)
        self._mux(wav)
        self._video_path.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)
        return self.path

    # -- internals ----------------------------------------------------------------------

    def _start(self, width: int, height: int) -> None:
        # yuv420p needs even sides; pad by one pixel if the window is odd.
        pad = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        self._proc = subprocess.Popen(
            [
                self._ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                "-r", str(self.fps), "-i", "-",
                "-vf", pad, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", str(self._video_path),
            ],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        self._thread = threading.Thread(target=self._write, name="demo-video", daemon=True)
        self._thread.start()

    def _write(self) -> None:
        """Place each frame on the constant-rate timeline; a gap repeats the last frame."""
        assert self._proc is not None and self._proc.stdin is not None
        out = self._proc.stdin
        last: bytes | None = None
        try:
            while (item := self._q.get()) is not None:
                t, image = item
                if self._t0 is None:
                    self._t0 = t
                index = round((t - self._t0) * self.fps)
                data = np.ascontiguousarray(image).tobytes()
                while last is not None and self._written < index:
                    out.write(last)
                    self._written += 1
                if self._written <= index:  # a frame that lands on an index already filled
                    out.write(data)  # is skipped, so the timeline never runs ahead
                    self._written += 1
                last = data
        except (BrokenPipeError, OSError) as exc:
            self.error = str(exc)

    def _mux(self, wav: Path) -> None:
        cmd = [
            self._ffmpeg, "-y", "-loglevel", "error",
            "-i", str(self._video_path), "-i", str(wav),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(self.path),
        ]  # fmt: skip
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise DemoError(f"ffmpeg failed while adding audio: {done.stderr.strip()}")
