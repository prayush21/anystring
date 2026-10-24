"""HandTracker tests with a fake landmarker: no model file, no network."""

from __future__ import annotations

import numpy as np
import pytest
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from racketguitar.config import FilterConfig, HandConfig
from racketguitar.vision.hands import (
    MODEL_URL,
    HandModelMissingError,
    HandTracker,
    resolve_model_path,
)
from tests.fakes import place_hand

W, H = 1280, 720


class FakeLandmarker:
    """Returns scripted hands (pixel arrays in the full frame) and records every call."""

    def __init__(self, script: list[list[np.ndarray]]):
        self.script = list(script)
        self.timestamps: list[int] = []
        self.images: list = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        assert isinstance(timestamp_ms, int)
        self.timestamps.append(timestamp_ms)
        self.images.append(image)
        hands = self.script.pop(0) if self.script else []
        return vision.HandLandmarkerResult(
            handedness=[[Category(score=0.9, category_name="Left")] for _ in hands],
            hand_landmarks=[
                [NormalizedLandmark(x=x / W, y=y / H, z=0.0) for x, y in px] for px in hands
            ],
            hand_world_landmarks=[],
        )

    def close(self):
        self.closed = True


def make_tracker(script, cfg: HandConfig | None = None, filters: FilterConfig | None = None):
    made = {}

    def factory(options):
        made["options"] = options
        made["lm"] = FakeLandmarker(script)
        return made["lm"]

    tracker = HandTracker(
        cfg or HandConfig(), filters or FilterConfig(), landmarker_factory=factory
    )
    return tracker, made["lm"], made["options"]


def frame(color=(255, 0, 0)) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = color
    return img


RIGHT = place_hand("pick", (900.0, 400.0), palm_px=90.0)
LEFT = place_hand("fist", (300.0, 450.0), palm_px=90.0, angle_deg=90.0)


def test_options_use_video_mode_and_config():
    cfg = HandConfig(num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.25)
    _, _, opts = make_tracker([], cfg)
    assert opts.running_mode == vision.RunningMode.VIDEO
    assert opts.num_hands == 2
    assert opts.min_hand_detection_confidence == 0.6
    assert opts.min_hand_presence_confidence == cfg.min_presence_confidence
    assert opts.min_tracking_confidence == 0.25
    assert opts.base_options.model_asset_path.endswith("hand_landmarker.task")


def test_timestamps_are_real_capture_times_in_ms():
    tracker, lm, _ = make_tracker([])
    # Irregular intervals, as a real 30..60 fps camera delivers them.
    times = [5000.0, 5000.0167, 5000.0501, 5000.0502, 5000.0503, 5000.1]
    for t in times:
        tracker.process(frame(), t)
    assert lm.timestamps == [5000000, 5000017, 5000050, 5000051, 5000052, 5000100]
    assert all(b > a for a, b in zip(lm.timestamps, lm.timestamps[1:], strict=False))


def test_same_frame_twice_is_not_redetected_and_backwards_raises():
    tracker, lm, _ = make_tracker([[RIGHT]])
    a = tracker.process(frame(), 1.0)
    b = tracker.process(frame(), 1.0)
    assert len(lm.timestamps) == 1
    assert b[0] is a[0]
    with pytest.raises(ValueError, match="backwards"):
        tracker.process(frame(), 0.5)


def test_detects_on_downscaled_rgb_copy():
    tracker, lm, _ = make_tracker([])
    tracker.process(frame(color=(255, 0, 0)), 1.0)  # pure blue in BGR
    img = lm.images[0]
    assert (img.width, img.height) == (640, 360)
    assert img.numpy_view()[0, 0].tolist() == [0, 0, 255]  # RGB


def test_landmarks_scaled_back_to_full_resolution():
    tracker, _, _ = make_tracker([[RIGHT, LEFT]])
    obs = tracker.process(frame(), 2.0)
    assert len(obs) == 2
    for o, truth in zip(obs, (RIGHT, LEFT), strict=True):
        assert o.t == 2.0
        assert o.raw.shape == (21, 2) and o.landmarks.shape == (21, 2)
        np.testing.assert_allclose(o.raw, truth, atol=1e-3)  # float32 in MediaPipe containers
        np.testing.assert_array_equal(o.landmarks, o.raw)  # first frame: filter passes through


def test_filter_reduces_jitter_but_raw_is_untouched(rng):
    n = 120
    noisy = [RIGHT + rng.normal(0, 2.0, RIGHT.shape) for _ in range(n)]
    tracker, _, _ = make_tracker([[x] for x in noisy])
    raw_err, filt_err = [], []
    for k in range(n):
        (o,) = tracker.process(frame(), 10.0 + k / 60)
        np.testing.assert_allclose(o.raw, noisy[k], atol=1e-3)
        if k > 30:
            raw_err.append(np.std(o.raw - RIGHT))
            filt_err.append(np.std(o.landmarks - RIGHT))
    assert np.mean(filt_err) < 0.5 * np.mean(raw_err)
    assert tracker.tracks[0].age == n - 1


def test_filter_state_follows_each_hand_when_order_swaps():
    step = np.array([3.0, 0.0])
    script = [[RIGHT + k * step, LEFT - k * step][:: 1 if k % 2 == 0 else -1] for k in range(10)]
    tracker, _, _ = make_tracker(script)
    ids = {}
    for k in range(10):
        obs = tracker.process(frame(), 1.0 + k / 60)
        for o, tr in zip(obs, tracker.tracks, strict=True):
            is_right = o.raw[0, 0] > W / 2
            ids.setdefault(is_right, tr.id)
            assert tr.id == ids[is_right]
            assert tr.age == k
            # Filtered landmarks stay with their own hand, never pulled toward the other.
            assert np.max(np.abs(o.landmarks - o.raw)) < 40.0
    assert len(set(ids.values())) == 2


def test_reacquired_hand_gets_a_fresh_filter():
    far = RIGHT + (60.0, 0.0)
    tracker, _, _ = make_tracker([[RIGHT], [RIGHT], [], [far]])
    tracker.process(frame(), 1.00)
    tracker.process(frame(), 1.02)
    first_id = tracker.tracks[0].id
    assert tracker.process(frame(), 1.04) == []
    assert tracker.tracks == []
    (o,) = tracker.process(frame(), 1.06)
    (tr,) = tracker.tracks
    assert tr.id != first_id and tr.age == 0
    np.testing.assert_array_equal(o.landmarks, o.raw)  # no glide from the stale position


def test_far_jump_is_a_new_hand():
    tracker, _, _ = make_tracker([[RIGHT], [RIGHT + (500.0, 0.0)]])
    tracker.process(frame(), 1.0)
    a = tracker.tracks[0].id
    tracker.process(frame(), 1.02)
    assert tracker.tracks[0].id != a and tracker.tracks[0].age == 0


def test_close_closes_landmarker():
    tracker, lm, _ = make_tracker([])
    with tracker:
        pass
    assert lm.closed


def test_missing_model_names_download_url(tmp_path):
    cfg = HandConfig(model_path=str(tmp_path / "nope" / "hand_landmarker.task"))
    with pytest.raises(HandModelMissingError) as err:
        HandTracker(cfg)
    assert MODEL_URL in str(err.value)
    assert isinstance(err.value, FileNotFoundError)


def test_resolve_model_path_finds_existing(tmp_path):
    p = tmp_path / "m.task"
    p.write_bytes(b"x")
    assert resolve_model_path(p) == p.resolve()


@pytest.mark.hardware
def test_real_model_runs_in_video_mode():
    cfg = HandConfig()
    try:
        resolve_model_path(cfg.model_path)
    except HandModelMissingError:
        pytest.skip("hand model not downloaded")
    with HandTracker(cfg) as tracker:
        for k in range(3):
            assert tracker.process(frame(color=(90, 90, 90)), 100.0 + k / 30) == []
