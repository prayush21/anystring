"""Replay a recorded session through the play logic, with no camera, mic or speakers.

By default the recorded observations (poses, hands, drained onsets) go straight back
into a fresh ``Pipeline``; the result must match the recorded notes exactly. Two
options re-run detectors on the recorded media instead:

* ``onsets_from="mic"``: stream ``mic.wav`` block by block through a fresh
  ``OnsetDetector``, as the mic callback does. Each onset goes to the first frame whose
  processing started after the block that reported it ended, as live draining would
  (live, the callback runs a little after the block ends, so this is the earliest case).
* ``markers_from="corners"``: re-run pose tracking (filters, hold) on the recorded raw
  marker detections; no video needed, so it is the cheap way to try marker settings.
* ``markers_from="video"``: re-detect markers on ``video.mkv`` (lossless, so detections
  match live); frames the writer dropped use their recorded raw detections. Hands stay
  as recorded: hand tracking needs the model file.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from racketguitar.audio.output import render_offline
from racketguitar.audio.synth import make_synth
from racketguitar.capture.mic import OnsetDetector
from racketguitar.config import Calibration, Config
from racketguitar.pipeline import FrameInput, Pipeline
from racketguitar.replay.record import Session, load_session
from racketguitar.types import NoteEvent, Onset
from racketguitar.vision.markers import MarkerTracker, detect_markers

__all__ = [
    "ReplayResult",
    "replay",
    "compare_events",
    "render_session",
    "stream_onsets",
    "video_frames",
]


@dataclass(frozen=True, eq=False)
class ReplayResult:
    frames: list[FrameInput]  # the inputs actually fed (after any re-detection)
    events: list[NoteEvent]
    per_frame: list[list[NoteEvent]]
    mismatches: list[str]  # differences from the recorded notes; empty = identical

    @property
    def matches(self) -> bool:
        return not self.mismatches


def replay(
    session: Session | str | Path,
    *,
    config: Config | None = None,
    calibration: Calibration | None = None,
    onsets_from: Literal["recorded", "mic"] = "recorded",
    markers_from: Literal["recorded", "corners", "video"] = "recorded",
) -> ReplayResult:
    """Run ``session`` through a fresh pipeline. ``config``/``calibration`` override the
    recorded ones (to try new settings on an old take); then mismatches are expected."""
    s = load_session(session) if isinstance(session, (str, Path)) else session
    cfg = config or s.config
    frames = [f.inp for f in s.frames]
    if onsets_from == "mic":
        frames = _with_mic_onsets(frames, s, cfg)
    if markers_from != "recorded" and s.meta.get("pose_source") == "flow":
        raise ValueError("markerless session: only its recorded poses can be replayed")
    if markers_from != "recorded":
        frames = _retracked(frames, s, cfg, use_video=markers_from == "video")
    pipe = Pipeline(cfg, calibration or s.calibration)
    per_frame = pipe.run(frames)
    events = [e for evs in per_frame for e in evs]
    return ReplayResult(frames, events, per_frame, compare_events(s.events, events))


def compare_events(
    expected: list[NoteEvent], actual: list[NoteEvent], limit: int = 20
) -> list[str]:
    """Human-readable differences between two note lists (exact comparison)."""
    out = []
    for i, (a, b) in enumerate(zip(expected, actual, strict=False)):
        if a != b:
            out.append(f"event {i}: recorded {a} != replayed {b}")
            if len(out) >= limit:
                return out
    if len(expected) != len(actual):
        out.append(f"recorded {len(expected)} events, replayed {len(actual)}")
    return out


def render_session(events: list[NoteEvent], cfg: Config, *, tail_s: float = 2.0) -> np.ndarray:
    """What the synth would have played for ``events``, as a mono float32 array."""
    if not events:
        return np.zeros(0, dtype=np.float32)
    start = min(e.t_detect for e in events)
    dur = max(e.t_detect for e in events) - start + tail_s
    return render_offline(
        make_synth(cfg),
        events,
        dur,
        start_time=start,
        master_gain=cfg.audio.master_gain,
        clip_knee=cfg.audio.clip_knee,
    )


def video_frames(path: str | Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            yield img
    finally:
        cap.release()


def _with_mic_onsets(frames: list[FrameInput], s: Session, cfg: Config) -> list[FrameInput]:
    if s.mic is None or s.mic_t0 is None:
        raise ValueError("session has no mic recording")
    found = stream_onsets(s.mic, s.mic_t0, OnsetDetector(cfg.mic))
    out, k = [], 0
    for f in frames:
        batch: list[Onset] = []
        while k < len(found) and found[k][0] <= f.proc_t:
            batch.append(found[k][1])
            k += 1
        out.append(replace(f, onsets=tuple(batch)))
    return out


def stream_onsets(x: np.ndarray, t0: float, det: OnsetDetector) -> list[tuple[float, Onset]]:
    """``(reported_at, onset)`` for ``x`` fed in callback-sized blocks from time ``t0``;
    ``reported_at`` is the end time of the block that produced the onset."""
    bs, sr = det.cfg.block_size, det.cfg.sample_rate
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    a = np.pad(a, (0, (-a.size) % bs))
    out = []
    for i in range(0, a.size, bs):
        t_block = t0 + i / sr
        for o in det.process(a[i : i + bs], t_block):
            out.append((t_block + bs / sr, o))
    return out


def _retracked(
    frames: list[FrameInput], s: Session, cfg: Config, *, use_video: bool
) -> list[FrameInput]:
    """New poses from a fresh tracker fed video detections and/or recorded corners.

    A frame with neither (an old recording that dropped its image) keeps its recorded
    pose, and the tracker misses that frame, so poses after it may differ slightly.
    """
    if use_video and (s.video_path is None or not Path(s.video_path).exists()):
        raise ValueError("session has no video recording")
    if not use_video and any(r.corners is None for r in s.frames):
        raise ValueError("session has no recorded marker corners")
    tracker = MarkerTracker(cfg.markers, cfg.filters)
    ids = (cfg.markers.tip_id, cfg.markers.throat_id)
    images = video_frames(s.video_path) if use_video else iter(())
    out = []
    for f, rec in zip(frames, s.frames, strict=True):
        if use_video and rec.in_video:
            img = next(images, None)
            if img is None:
                raise ValueError("video has fewer frames than the session records")
            corners = detect_markers(tracker.detector, img, ids)
        elif rec.corners is not None:
            corners = rec.corners
        else:
            out.append(f)
            continue
        out.append(replace(f, pose=tracker.update_corners(corners, f.t)))
    return out
