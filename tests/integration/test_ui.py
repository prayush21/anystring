"""Overlay drawing and the calibration flow, headless."""

from __future__ import annotations

import numpy as np
import pytest

from racketguitar.config import Calibration, Config
from racketguitar.pipeline import FrameInput, Pipeline
from racketguitar.replay.record import TeeDetector
from racketguitar.ui.calibrate import BRUSH_S, GRIP_CAPTURE_S, QUIET_S, SETTLE_S, CalibrationFlow
from racketguitar.ui.overlay import BGR, CalibrationView, Overlay, Status, View
from tests.fakes import (
    brush_strokes,
    hand_obs,
    make_pose,
    mix,
    place_hand,
    render_marker_scene,
    room_noise,
    uv_to_px,
)
from tests.fakes.hands import INDEX_TIP
from tests.integration.song import frames


def _snapshot_mid_strum(cfg):
    """Run the song until the first notes, return (frame, snapshot, pipeline)."""
    song, _ = frames(cfg)
    pipe = Pipeline(cfg)
    for f in song:
        if any(e.kind == "on" for e in pipe.step(FrameInput(f.t, f.t, f.pose, f.hands, ()))):
            return f, pipe.snapshot()
    raise AssertionError("song never played")


def _mask(img: np.ndarray, colour: tuple[int, int, int], tol: int = 40) -> np.ndarray:
    return np.all(np.abs(img.astype(int) - colour) <= tol, axis=-1)


@pytest.mark.parametrize("mirror", [True, False])
def test_overlay_draws_strings_where_the_detectors_look(cfg, mirror):
    cfg = cfg.with_overrides({"ui": {"mirror": mirror}})
    f, snap = _snapshot_mid_strum(cfg)
    image = render_marker_scene(**f.scene).image
    view = View(image, f.t, f.pose, {}, snap, Status(fps=60.0, latency_ms=18.0))
    out = Overlay(cfg).draw(view)
    assert out.shape == image.shape and out is not image

    # A flashed (down-strum) string is drawn in the accent colour along its segment.
    hit = next(r for r in snap.recent if r.played)
    seg = snap.strings_px()[hit.hit.string]
    mid = seg.mean(axis=0)
    x = image.shape[1] - 1 - mid[0] if mirror else mid[0]
    y0, y1 = int(mid[1]) - 6, int(mid[1]) + 7
    x0, x1 = int(x) - 6, int(x) + 7
    patch = out[y0:y1, x0:x1]
    accent = _mask(patch, BGR["accent"], tol=90)
    assert accent.any(), "flashed string not drawn where the engine put it"


def test_overlay_handles_lost_racket_and_panels(cfg):
    pipe = Pipeline(cfg)
    pose = make_pose(visible=False)
    pipe.step(FrameInput(0.0, 0.0, pose, (), ()))
    img = np.full((720, 1280, 3), 100, np.uint8)
    status = Status(mic_level=0.01, onsets=(0.95,), recording=True, message="hello")
    cal = CalibrationView("Show both racket markers", 0.5, warning="careful")
    out = Overlay(cfg).draw(View(img, 1.0, pose, {}, pipe.snapshot(), status, cal))
    assert out.shape == img.shape
    assert _mask(out, BGR["rec"], tol=10).any()  # REC dot
    assert _mask(out[150:163], BGR["accent"], tol=10).any()  # calibration progress bar
    assert _mask(out, BGR["lost"], tol=10).any()  # "marker lost" text


def test_overlay_marks_found_markers(cfg):
    scene = render_marker_scene()
    pipe = Pipeline(cfg)
    pipe.step(FrameInput(0.0, 0.0, scene.pose, (), ()))
    view = View(scene.image, 0.0, scene.pose, scene.corners, pipe.snapshot())
    out = Overlay(cfg).draw(view)
    assert _mask(out, BGR["ok"], tol=10).sum() > 100


# -- calibration flow -------------------------------------------------------------------


def _hand(pose, shape, t):
    tip = uv_to_px(pose, 0.5, 0.45)
    return hand_obs(t, place_hand(shape, tip, anchor=INDEX_TIP, palm_px=80.0))


def _drive(flow, pose, shape, t, dur, fps=30.0):
    n = int(dur * fps) + 1
    for k in range(n):
        tk = t + k / fps
        flow.update(tk, pose, None if shape is None else _hand(pose, shape, tk))
    return t + n / fps


def test_calibration_flow_records_grip_scores():
    cfg = Config()
    flow = CalibrationFlow(cfg, Calibration(mic_noise_floor=1e-4))
    pose = make_pose()
    t = _drive(flow, make_pose(visible=False), None, 0.0, 2.0)
    assert flow.step == "markers" and flow.progress == 0.0
    t = _drive(flow, pose, None, t, 1.1)
    assert flow.step == "closed"
    t = _drive(flow, pose, None, t, 1.0)  # no hand: waits
    assert flow.step == "closed" and flow.progress == 0.0
    t = _drive(flow, pose, "pick", t, SETTLE_S + GRIP_CAPTURE_S + 0.1)
    assert flow.step == "open"
    _drive(flow, pose, "open", t, SETTLE_S + GRIP_CAPTURE_S + 0.1)
    assert flow.done and flow.error is None
    cal = flow.result
    assert cal is not None and cal.grip_closed_score < cal.grip_open_score
    assert cal.mic_noise_floor == 1e-4  # untouched fields come from the base


def test_calibration_flow_restarts_when_grips_are_swapped():
    cfg = Config()
    flow = CalibrationFlow(cfg)
    pose = make_pose()
    t = _drive(flow, pose, None, 0.0, 1.1)
    t = _drive(flow, pose, "open", t, SETTLE_S + GRIP_CAPTURE_S + 0.1)  # wrong: open first
    t = _drive(flow, pose, "pick", t, SETTLE_S + GRIP_CAPTURE_S + 0.1)
    assert flow.step == "closed" and flow.error is not None
    t = _drive(flow, pose, "pick", t, SETTLE_S + GRIP_CAPTURE_S + 0.1)
    _drive(flow, pose, "open", t, SETTLE_S + GRIP_CAPTURE_S + 0.1)
    assert flow.done and flow.error is None


def test_calibration_flow_measures_the_mic():
    cfg = Config()
    tee = TeeDetector(cfg.mic, seconds=10.0)
    flow = CalibrationFlow(cfg, mic=tee)
    pose = make_pose()
    bs, sr = cfg.mic.block_size, cfg.mic.sample_rate
    noise = room_noise(QUIET_S + 0.5, rms=0.003)
    brushes, _ = brush_strokes([0.5, 1.2, 1.9, 2.6, 3.3], BRUSH_S + 0.5, peak=0.1)
    audio = np.concatenate([noise, mix(brushes, room_noise(BRUSH_S + 0.5, rms=0.003))])
    fed = 0

    def feed_until(t):  # the mic callback keeps delivering blocks in real time
        nonlocal fed
        while fed + bs <= audio.size and (fed + bs) / sr <= t:
            tee.process(audio[fed : fed + bs], fed / sr)
            fed += bs

    t = 0.0
    for shape, dur in ((None, 1.1), ("pick", 2.6), ("open", 2.6)):
        t = _drive(flow, pose, shape, t, dur)
    assert flow.step == "quiet"
    start = fed = round(t * sr) // bs * bs  # quiet audio starts now
    audio = np.concatenate([np.zeros(start, np.float32), audio])
    while not flow.done:
        feed_until(t)
        flow.update(t, pose, None)
        t += 1 / 30
    cal = flow.result
    assert cal.mic_noise_floor is not None and cal.mic_brush_level is not None
    assert cal.mic_brush_level > 3 * cal.mic_noise_floor
    assert flow.warning is None


def test_text_outline_does_not_trail_past_the_text():
    import cv2

    from racketguitar.ui.overlay import FONT, _text

    img = np.zeros((60, 600, 3), np.uint8)
    s = "30 fps  vision 36.4 ms  latency -- ms"
    _text(img, s, (10, 40), 0.55, (255, 255, 255))
    (w, _), _ = cv2.getTextSize(s, FONT, 0.55, 1)
    cols = np.flatnonzero(img.any(axis=(0, 2)))
    assert cols.max() <= 10 + w + 2


def test_calibration_flow_rejects_grips_that_look_alike():
    """A 'spread' hand that scores barely above the pick grip (e.g. still half closed)
    must not be saved: it would put the arming threshold above everything."""
    cfg = Config()
    flow = CalibrationFlow(cfg)
    pose = make_pose()
    t = _drive(flow, pose, None, 0.0, 1.1)
    t = _drive(flow, pose, "pick", t, SETTLE_S + GRIP_CAPTURE_S + 0.1)
    _drive(flow, pose, "fist", t, SETTLE_S + GRIP_CAPTURE_S + 0.1)  # not opened
    assert flow.step == "closed" and flow.error is not None and "too alike" in flow.error
    assert flow.result is None
