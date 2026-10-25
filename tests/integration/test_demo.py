"""Demo capture: window frames plus rendered guitar sound end up in one MP4."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from anystring.replay.demo import DemoRecorder
from anystring.types import NoteEvent

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

T0 = 100.0


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def test_video_and_audio_streams_with_matching_length(cfg, tmp_path):
    rec = DemoRecorder(tmp_path / "out" / "demo.mp4", cfg, fps=30, drop_frames=False)
    rec.add_events([NoteEvent(T0 + 0.5, "on", 0, 40, 100), NoteEvent(T0 + 1.0, "on", 1, 45, 90)])
    for i in range(61):  # 2 s at 30 fps, odd width to exercise the padding
        img = np.full((90, 161, 3), i * 4, np.uint8)
        rec.add_frame(T0 + i / 30, img)
    path = rec.stop()
    assert path == tmp_path / "out" / "demo.mp4"
    info = probe(path)
    kinds = {s["codec_type"] for s in info["streams"]}
    assert kinds == {"video", "audio"}
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (162, 90)
    assert float(info["format"]["duration"]) == pytest.approx(61 / 30, abs=0.15)
    assert not list(path.parent.glob("*.video.mp4")) and not list(path.parent.glob("*.wav"))


def test_gap_in_frames_keeps_the_timeline(cfg, tmp_path):
    rec = DemoRecorder(tmp_path / "demo.mp4", cfg, fps=30, drop_frames=False)
    img = np.zeros((64, 64, 3), np.uint8)
    for t in (0.0, 1 / 30, 1.0):  # one second with no window update
        rec.add_frame(T0 + t, img)
    path = rec.stop()
    assert float(probe(path)["format"]["duration"]) == pytest.approx(31 / 30, abs=0.15)


def test_nothing_captured_returns_none(cfg, tmp_path):
    assert DemoRecorder(tmp_path / "demo.mp4", cfg).stop() is None
