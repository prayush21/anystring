"""MediaPipe Hand Landmarker wrapper -> ``list[HandObs]`` per frame.

Runs the landmarker in ``RunningMode.VIDEO`` with ``detect_for_video``, which is
synchronous and uses the previous frame's result to track, so each frame's landmarks
belong to that frame. Timestamps are the real capture times in integer milliseconds
(bumped by 1 ms only if two distinct frames round to the same millisecond, because
MediaPipe requires strictly increasing timestamps).

Detection runs on a downscaled copy (``HandConfig.detect_width x detect_height``);
landmarks come back normalised, so they are scaled to full-resolution pixels exactly.
Each hand keeps a One Euro filter across frames, matched to the previous frame's hands
by landmark distance. A hand that is lost and re-acquired starts a fresh filter.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from racketguitar.config import FilterConfig, HandConfig
from racketguitar.types import NUM_LANDMARKS, HandObs

from .filters import OneEuroFilter, fingertip_filter

__all__ = [
    "MODEL_URL",
    "HandModelMissingError",
    "HandTrack",
    "HandTracker",
    "build_options",
    "resolve_model_path",
]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
WRIST, MIDDLE_MCP = 0, 9

_REPO_ROOT = Path(__file__).resolve().parents[2]


class HandLandmarkerLike(Protocol):
    """The part of ``mediapipe.tasks.python.vision.HandLandmarker`` used here."""

    def detect_for_video(self, image: mp.Image, timestamp_ms: int) -> Any: ...
    def close(self) -> None: ...


LandmarkerFactory = Callable[[vision.HandLandmarkerOptions], HandLandmarkerLike]


class HandModelMissingError(FileNotFoundError):
    pass


def resolve_model_path(model_path: str | Path) -> Path:
    """Find the model file (as given, then relative to the repo root) or raise."""
    p = Path(model_path).expanduser()
    candidates = [p] if p.is_absolute() else [p, _REPO_ROOT / p]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise HandModelMissingError(
        f"hand landmarker model not found at {model_path!r}. Download it with:\n"
        f"  mkdir -p models && curl -L -o models/hand_landmarker.task {MODEL_URL}"
    )


def build_options(cfg: HandConfig, model_path: str | Path) -> vision.HandLandmarkerOptions:
    return vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=cfg.num_hands,
        min_hand_detection_confidence=cfg.min_detection_confidence,
        min_hand_presence_confidence=cfg.min_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )


def _default_factory(options: vision.HandLandmarkerOptions) -> HandLandmarkerLike:
    return vision.HandLandmarker.create_from_options(options)


@dataclass(frozen=True)
class HandTrack:
    """Identity of one emitted hand, aligned with the ``HandObs`` list of a frame."""

    id: int  # stable while the hand stays tracked; a re-acquired hand gets a new id
    age: int  # frames since acquired: 0 on the first frame (filter just reset)
    handedness: str | None  # MediaPipe's label; it assumes a mirrored image, so on our
    # un-mirrored frames "Left" is the person's right hand


@dataclass
class _Track:
    id: int
    filt: OneEuroFilter
    raw: np.ndarray
    age: int = 0


class HandTracker:
    """Per-frame hand landmarks with per-hand One Euro filtering.

    ``landmarker_factory`` receives the ``HandLandmarkerOptions`` and returns an object
    with ``detect_for_video``; inject a fake in tests. Without one, the model file at
    ``HandConfig.model_path`` must exist. Not thread-safe: call ``process`` from the
    vision thread with increasing capture times.
    """

    def __init__(
        self,
        hands: HandConfig | None = None,
        filters: FilterConfig | None = None,
        *,
        landmarker_factory: LandmarkerFactory | None = None,
    ) -> None:
        self.cfg = hands or HandConfig()
        self.filter_cfg = filters or FilterConfig()
        if landmarker_factory is None:
            model = resolve_model_path(self.cfg.model_path)
            landmarker_factory = _default_factory
        else:
            model = Path(self.cfg.model_path)
        self.options = build_options(self.cfg, model)
        self._landmarker = landmarker_factory(self.options)
        self._ids = itertools.count()
        self._tracks: list[_Track] = []
        self._last_t: float | None = None
        self._last_ts_ms: int | None = None
        self._last_out: list[HandObs] = []
        self._last_info: list[HandTrack] = []

    @property
    def tracks(self) -> list[HandTrack]:
        """Track info for the hands returned by the latest ``process`` call, same order."""
        return list(self._last_info)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def timestamp_ms(self, t: float) -> int:
        """Strictly increasing integer ms for MediaPipe from capture time ``t``."""
        ts = round(t * 1000.0)
        if self._last_ts_ms is not None and ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        return ts

    def process(self, image: np.ndarray, t: float) -> list[HandObs]:
        """Detect hands in a full-resolution BGR frame captured at ``t``."""
        if self._last_t is not None and t <= self._last_t:
            if t == self._last_t:  # same frame handed in twice
                return list(self._last_out)
            raise ValueError(f"capture time went backwards: {t} after {self._last_t}")
        h, w = image.shape[:2]
        small = image
        if (w, h) != (self.cfg.detect_width, self.cfg.detect_height):
            small = cv2.resize(
                image, (self.cfg.detect_width, self.cfg.detect_height), interpolation=cv2.INTER_AREA
            )
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ts = self.timestamp_ms(t)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
        )
        self._last_t, self._last_ts_ms = t, ts

        raws, labels = _parse_result(result, w, h)
        out, info = self._associate(raws, labels, t)
        self._last_out, self._last_info = out, info
        return list(out)

    def _associate(
        self, raws: list[np.ndarray], labels: list[str | None], t: float
    ) -> tuple[list[HandObs], list[HandTrack]]:
        # Greedy nearest matching of detections to last frame's hands.
        pairs = []
        for di, raw in enumerate(raws):
            for ti, tr in enumerate(self._tracks):
                palm = max(float(np.hypot(*(tr.raw[MIDDLE_MCP] - tr.raw[WRIST]))), 1.0)
                d = float(np.mean(np.hypot(*(raw - tr.raw).T)))
                if d < self.cfg.assoc_max_palms * palm:
                    pairs.append((d, di, ti))
        pairs.sort()
        match: dict[int, _Track] = {}
        used: set[int] = set()
        for _, di, ti in pairs:
            if di not in match and ti not in used:
                match[di] = self._tracks[ti]
                used.add(ti)

        tracks, out, info = [], [], []
        for di, raw in enumerate(raws):
            tr = match.get(di)
            if tr is None:
                tr = _Track(id=next(self._ids), filt=fingertip_filter(self.filter_cfg), raw=raw)
            else:
                tr.age += 1
                tr.raw = raw
            filtered = tr.filt(raw, t)
            tracks.append(tr)
            out.append(HandObs(t=float(t), landmarks=filtered, raw=raw.copy()))
            info.append(HandTrack(id=tr.id, age=tr.age, handedness=labels[di]))
        self._tracks = tracks  # unmatched old tracks are dropped: re-acquired = new filter
        return out, info


def _parse_result(result: Any, width: int, height: int) -> tuple[list[np.ndarray], list]:
    """Normalised MediaPipe landmarks -> full-resolution pixel arrays and labels."""
    hands: Sequence = getattr(result, "hand_landmarks", None) or []
    handedness: Sequence = getattr(result, "handedness", None) or []
    raws, labels = [], []
    for i, lms in enumerate(hands):
        if len(lms) != NUM_LANDMARKS:
            continue
        raws.append(np.array([(lm.x * width, lm.y * height) for lm in lms], dtype=float))
        cats = handedness[i] if i < len(handedness) else []
        labels.append(cats[0].category_name if cats else None)
    return raws, labels
