"""Camera thread tests with a fake capture object: no real camera, no sleeps.

The fake's ``grab()`` blocks until the test releases a frame, so the test controls
exactly how many frames the thread has captured. Because the thread stores each frame
before calling ``grab()`` again, "grab entered n+1 times" means frames 0..n-1 are stored.
"""

from __future__ import annotations

import itertools
import sys
import threading

import cv2
import numpy as np
import pytest

from racketguitar.capture.camera import Camera, CameraError
from racketguitar.config import CameraConfig

WAIT = 2.0  # generous timeout for thread hand-offs; never reached when things work


class FakeCapture:
    def __init__(self, *, max_fps: float = 60.0, opened: bool = True, fail: bool = False):
        self.max_fps = max_fps
        self.opened = opened
        self.fail = fail
        self.props: dict[int, float] = {}
        self.set_calls: list[tuple[int, float]] = []
        self._frames = threading.Semaphore(0)
        self._closed = threading.Event()
        self._grabs = 0
        self._grab_cond = threading.Condition()
        self.served = 0
        self.released = False

    # VideoCapture API
    def isOpened(self) -> bool:  # noqa: N802
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_FPS:
            self.props[prop] = min(value, self.max_fps)
            return value <= self.max_fps
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0.0)

    def grab(self) -> bool:
        with self._grab_cond:
            self._grabs += 1
            self._grab_cond.notify_all()
        if self.fail:
            return False
        while not self._frames.acquire(timeout=0.01):
            if self._closed.is_set():
                return False
        return True

    def retrieve(self):
        img = np.full((4, 6, 3), self.served % 256, np.uint8)
        self.served += 1
        return True, img

    def release(self) -> None:
        self.released = True

    # test controls
    def push(self, n: int = 1) -> None:
        for _ in range(n):
            self._frames.release()

    def wait_grabs(self, n: int) -> None:
        with self._grab_cond:
            assert self._grab_cond.wait_for(lambda: self._grabs >= n, WAIT)

    def close(self) -> None:
        self._closed.set()


def counter_clock(start: float = 100.0, fps: float = 60.0):
    ticks = itertools.count()
    return lambda: start + next(ticks) / fps


@pytest.fixture
def make_camera():
    made: list[tuple[Camera, FakeCapture]] = []

    def factory(fake: FakeCapture | None = None, cfg: CameraConfig | None = None, **kw):
        fake = fake or FakeCapture()
        calls = []

        def capture_factory(device, backend):
            calls.append((device, backend))
            return fake

        cam = Camera(cfg or CameraConfig(), capture_factory=capture_factory, **kw)
        cam.factory_calls = calls
        made.append((cam, fake))
        return cam, fake

    yield factory
    for cam, fake in made:
        fake.close()
        cam.stop()


def test_configures_capture_and_gets_60fps(make_camera):
    cfg = CameraConfig(device=2, width=1280, height=720)
    cam, fake = make_camera(cfg=cfg)
    cam.start()
    assert cam.factory_calls == [(2, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else 0)]
    assert fake.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert fake.props[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert fake.props[cv2.CAP_PROP_BUFFERSIZE] == 1
    assert cam.fps == 60.0
    assert (cam.width, cam.height) == (1280, 720)


def test_falls_back_to_30fps(make_camera):
    cam, fake = make_camera(FakeCapture(max_fps=30.0))
    cam.start()
    fps_requests = [v for p, v in fake.set_calls if p == cv2.CAP_PROP_FPS]
    assert fps_requests == [60, 30]
    assert cam.fps == 30.0


def test_frames_carry_capture_time_and_index(make_camera):
    cam, fake = make_camera(clock=counter_clock())
    cam.start()
    for k in range(3):
        fake.push()
        frame = cam.read(timeout=WAIT)
        assert frame is not None
        assert frame.index == k
        assert frame.t == pytest.approx(100.0 + k / 60)
        assert frame.image[0, 0, 0] == k


def test_keeps_only_newest_frame(make_camera):
    cam, fake = make_camera(clock=counter_clock())
    cam.start()
    fake.push(5)
    fake.wait_grabs(6)  # frames 0..4 captured and stored, thread now waiting on the 6th
    frame = cam.read(timeout=WAIT)
    assert frame is not None and frame.index == 4
    assert cam.read(timeout=0.01) is None  # nothing newer yet; no stale backlog
    assert cam.latest() is frame
    fake.push()
    assert cam.read(timeout=WAIT).index == 5


def test_measured_fps_from_capture_times(make_camera):
    cam, fake = make_camera(clock=counter_clock(fps=60.0))
    cam.start()
    fake.push(20)
    fake.wait_grabs(21)
    assert cam.measured_fps == pytest.approx(60.0)


def test_open_failure_is_clear(make_camera):
    cam, fake = make_camera(FakeCapture(opened=False))
    with pytest.raises(CameraError, match="could not open camera"):
        cam.start()
    assert fake.released


def test_dead_camera_surfaces_error(make_camera):
    cam, _ = make_camera(FakeCapture(fail=True))
    cam.start()
    with pytest.raises(CameraError):
        cam.read(timeout=WAIT)


def test_stop_releases_and_unblocks_reader(make_camera):
    cam, fake = make_camera()
    cam.start()
    assert cam.running
    fake.close()
    cam.stop()
    assert not cam.running
    assert fake.released
    assert cam.read(timeout=WAIT) is None


def test_context_manager(make_camera):
    cam, fake = make_camera()
    with cam as c:
        assert c is cam and cam.running
        fake.close()
    assert fake.released and not cam.running


@pytest.mark.hardware
def test_real_camera_delivers_frames():
    with Camera(CameraConfig()) as cam:
        frames = [cam.read(timeout=WAIT) for _ in range(10)]
    assert all(f is not None for f in frames)
    assert all(b.t > a.t and b.index > a.index for a, b in zip(frames, frames[1:], strict=False))
    assert frames[-1].image.shape == (cam.height, cam.width, 3)
    assert cam.fps in (60.0, 30.0)
