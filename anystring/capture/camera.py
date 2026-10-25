"""Camera thread: OpenCV capture that keeps only the newest frame and its capture time.

A background thread grabs frames as fast as the camera delivers them, stamps each with
``time.monotonic()`` as soon as ``grab()`` returns (the closest the app gets to the
capture instant), and overwrites a single slot. Consumers never see a stale backlog:
``read()`` returns the newest frame they have not had yet.

The capture object is created by an injectable factory, so tests run with a fake.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from anystring.config import CameraConfig

__all__ = ["Frame", "Camera", "CameraError", "VideoCaptureLike", "default_capture_factory"]

FPS_TOLERANCE = 1.0  # a reported rate within this of the request counts as granted
MAX_CONSECUTIVE_FAILURES = 30  # grab() failures in a row before the thread gives up
FAILURE_BACKOFF_S = 0.005


class VideoCaptureLike(Protocol):
    """The part of ``cv2.VideoCapture`` the camera thread uses."""

    def isOpened(self) -> bool: ...  # noqa: N802 (OpenCV naming)
    def set(self, prop: int, value: float) -> bool: ...
    def get(self, prop: int) -> float: ...
    def grab(self) -> bool: ...
    def retrieve(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[int, int], VideoCaptureLike]


def default_backend() -> int:
    return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY


def default_capture_factory(device: int, backend: int) -> VideoCaptureLike:
    return cv2.VideoCapture(device, backend)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # (H, W, 3) uint8 BGR, full resolution, un-mirrored
    t: float  # capture time, time.monotonic() seconds
    index: int  # 0, 1, 2, ... in capture order; gaps mean frames were overwritten


class CameraError(RuntimeError):
    pass


class Camera:
    """Newest-frame-only capture on a background thread.

    Use as a context manager or call ``start()`` / ``stop()``. ``fps`` is the rate the
    driver agreed to (``CameraConfig.fps``, else ``fallback_fps``); ``measured_fps`` is
    what actually arrives.
    """

    def __init__(
        self,
        cfg: CameraConfig | None = None,
        *,
        capture_factory: CaptureFactory = default_capture_factory,
        backend: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg or CameraConfig()
        self._factory = capture_factory
        self._backend = default_backend() if backend is None else backend
        self._clock = clock
        self._cap: VideoCaptureLike | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._last_read = -1
        self._error: BaseException | None = None
        self._dt_ema: float | None = None
        self.fps: float = 0.0
        self.width = 0
        self.height = 0

    # -- lifecycle -------------------------------------------------------------------

    def start(self) -> Camera:
        if self._thread is not None:
            return self
        cap = self._factory(self.cfg.device, self._backend)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"could not open camera {self.cfg.device}; on macOS, grant camera access "
                "to your terminal in System Settings > Privacy & Security > Camera"
            )
        self._configure(cap)
        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._cond:
            self._cond.notify_all()

    def __enter__(self) -> Camera:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def measured_fps(self) -> float:
        """Smoothed rate of arriving frames, 0 until two frames have arrived."""
        dt = self._dt_ema
        return 1.0 / dt if dt else 0.0

    # -- consumers ---------------------------------------------------------------------

    def latest(self) -> Frame | None:
        """The newest frame, even if already returned by ``read``; None before the first."""
        with self._cond:
            return self._frame

    def read(self, timeout: float | None = None) -> Frame | None:
        """Block until a frame newer than the last one read arrives, then return it.

        Returns None on timeout or after ``stop()``. Raises ``CameraError`` if the capture
        thread died.
        """
        with self._cond:
            ok = self._cond.wait_for(self._fresh_or_done, timeout)
            if self._error is not None:
                raise CameraError("camera thread stopped") from self._error
            if not ok or self._frame is None or self._frame.index <= self._last_read:
                return None
            self._last_read = self._frame.index
            return self._frame

    def _fresh_or_done(self) -> bool:
        fresh = self._frame is not None and self._frame.index > self._last_read
        return fresh or self._stop.is_set() or self._error is not None

    # -- capture thread ----------------------------------------------------------------

    def _configure(self, cap: VideoCaptureLike) -> None:
        cfg = self.cfg
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = float(cfg.fallback_fps)
        for want in (cfg.fps, cfg.fallback_fps):
            accepted = cap.set(cv2.CAP_PROP_FPS, want)
            got = cap.get(cv2.CAP_PROP_FPS)
            if accepted and abs(got - want) <= FPS_TOLERANCE:
                self.fps = float(want)
                break
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cfg.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cfg.height

    def _run(self) -> None:
        cap = self._cap
        assert cap is not None
        index, failures, last_t = 0, 0, None
        try:
            while not self._stop.is_set():
                ok = cap.grab()
                t = self._clock()
                image = None
                if ok:
                    ok, image = cap.retrieve()
                if not ok or image is None:
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        raise CameraError(f"{failures} frame reads failed in a row")
                    self._stop.wait(FAILURE_BACKOFF_S)
                    continue
                failures = 0
                if last_t is not None and t > last_t:
                    dt = t - last_t
                    self._dt_ema = dt if self._dt_ema is None else 0.9 * self._dt_ema + 0.1 * dt
                last_t = t
                with self._cond:
                    self._frame = Frame(image=image, t=t, index=index)
                    self._cond.notify_all()
                index += 1
        except BaseException as e:  # surfaced to the consumer by read()
            with self._cond:
                self._error = e
                self._cond.notify_all()
