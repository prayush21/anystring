"""Clip snapping: F8 saves the last ``clip.seconds`` of the play window and its sound.

The display loop hands every window image to ``ClipBuffer.add_frame``, which keeps it as a
JPEG in a ring of the last ``seconds`` (raw 720p frames would be gigabytes). The play step
hands over its note events. ``snap`` copies the ring and encodes it on a worker thread with
``DemoRecorder``, so the picture is the window as drawn and the sound is the notes rendered
offline by the same synth, exactly as for ``--demo``. Nothing here is on the play path: the
window thread pays one JPEG encode (about 2 ms) per drawn frame, and the save runs apart.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from anystring.config import Config
from anystring.replay.demo import DemoRecorder, ffmpeg_path
from anystring.types import NoteEvent

__all__ = ["ClipBuffer", "ClipSave"]


@dataclass
class ClipSave:
    """One clip being written. The worker updates it; the UI only reads it.

    ``error`` is set before ``finished``, so a reader that sees ``finished`` sees the outcome.
    """

    path: Path
    seconds: float  # length of the clip
    frames: int
    started: float  # clock time of the F8 press
    done: int = 0  # frames handed to the encoder so far
    error: str | None = None
    finished: float | None = None  # clock time the save ended, ok or not

    @property
    def progress(self) -> float:
        """0..1; held under 1 while the audio is rendered and muxed after the last frame."""
        if self.finished is not None:
            return 1.0
        return min(self.done / max(1, self.frames), 0.95)


class ClipBuffer:
    def __init__(self, cfg: Config, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.clock = clock
        self.current: ClipSave | None = None  # the latest save, finished or not
        self._lock = threading.Lock()
        self._frames: deque[tuple[float, bytes]] = deque()
        self._events: deque[NoteEvent] = deque()
        self._thread: threading.Thread | None = None

    # -- feeding (window thread, vision thread) -------------------------------------------

    def add_frame(self, t: float, image: np.ndarray) -> None:
        """One window image taken at capture time ``t``."""
        ok, data = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.clip.jpeg_quality]
        )
        if not ok:
            return
        with self._lock:
            self._frames.append((t, data.tobytes()))
            cutoff = t - self.cfg.clip.seconds
            while self._frames and self._frames[0][0] < cutoff:
                self._frames.popleft()

    def add_events(self, events: Sequence[NoteEvent]) -> None:
        if not events:
            return
        with self._lock:
            self._events.extend(events)
            cutoff = events[-1].t_detect - self.cfg.clip.seconds
            while self._events and self._events[0].t_detect < cutoff:
                self._events.popleft()

    # -- saving ---------------------------------------------------------------------------

    @property
    def saving(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def buffered_s(self) -> float:
        with self._lock:
            return self._frames[-1][0] - self._frames[0][0] if len(self._frames) > 1 else 0.0

    def snap(self, folder: Path) -> ClipSave | None:
        """Start saving what is buffered to ``folder`` and return its progress record.

        None if no frame is buffered yet. Raises ``DemoError`` if ffmpeg is missing. The
        caller checks ``saving`` first: one save at a time.
        """
        ffmpeg_path()
        with self._lock:
            frames, events = list(self._frames), list(self._events)
        if not frames:
            return None
        span = frames[-1][0] - frames[0][0]
        save = ClipSave(
            path=_free_path(folder),
            seconds=span + 1.0 / 30.0,
            frames=len(frames),
            started=self.clock(),
        )
        self.current = save
        self._thread = threading.Thread(
            target=self._write, args=(save, frames, events), name="clip-save", daemon=True
        )
        self._thread.start()
        return save

    def wait(self, timeout: float | None = None) -> bool:
        """Block until a running save ends. True if none is running afterwards."""
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.saving

    def _write(
        self, save: ClipSave, frames: list[tuple[float, bytes]], events: list[NoteEvent]
    ) -> None:
        try:
            rec = DemoRecorder(save.path, self.cfg, drop_frames=False)
            # Notes from before the first frame would all land at sample 0 of the audio.
            rec.add_events([e for e in events if e.t_detect >= frames[0][0]])
            for t, data in frames:
                rec.add_frame(t, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
                save.done += 1
            rec.stop()
        except Exception as exc:  # DemoError, OSError: shown on the card, never raised
            save.error = str(exc) or type(exc).__name__
            for leftover in (
                save.path,
                save.path.with_name(save.path.stem + ".video.mp4"),
                save.path.with_name(save.path.stem + ".audio.wav"),
            ):
                leftover.unlink(missing_ok=True)
        finally:
            save.finished = self.clock()


def _free_path(folder: Path) -> Path:
    stem = f"clip-{time.strftime('%Y%m%d-%H%M%S')}"
    path = folder / f"{stem}.mp4"
    n = 2
    while path.exists():
        path = folder / f"{stem}-{n}.mp4"
        n += 1
    return path
