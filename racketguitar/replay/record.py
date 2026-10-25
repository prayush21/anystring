"""Session recording: per-frame observations, onsets, notes, mic audio and video.

A session is a directory::

    session.json      format version, timing, what was recorded
    config.toml       the settings the session ran with
    calibration.toml  the calibration it ran with
    data.npz          frame times, poses, hands, onsets and note events (float64)
    mic.wav           raw mic input, float32 (if the mic was running)
    video.mkv         lossless FFV1 camera frames, un-mirrored, for inspection (only
                      with ``[replay] video = true``)

Everything the play logic consumed is stored exactly, so ``replay.harness`` can feed it
back through the same ``Pipeline`` and must get the same notes.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.io import wavfile

from racketguitar.capture.mic import OnsetDetector
from racketguitar.config import (
    Calibration,
    Config,
    MicConfig,
    load_calibration,
    save_calibration,
    save_config,
)
from racketguitar.pipeline import FrameInput
from racketguitar.types import NUM_LANDMARKS, HandObs, NoteEvent, Onset, RacketPose, Vec2

__all__ = [
    "FORMAT_VERSION",
    "FrameRecord",
    "Session",
    "Recorder",
    "TeeDetector",
    "save_session",
    "load_session",
]

FORMAT_VERSION = 1
VIDEO_FOURCC = "FFV1"  # lossless: frames are exactly what the detectors saw
# Settings that old sessions recorded but this version no longer has (the marker
# tracker was removed); dropped on load so old recordings still replay.
_RETIRED = {"markers": None, "filters": ("marker_min_cutoff", "marker_beta")}
_KINDS = ("on", "damp", "all_off")


@dataclass(frozen=True, eq=False)
class FrameRecord:
    inp: FrameInput
    events: tuple[NoteEvent, ...]
    in_video: bool = False  # this frame's image is the next one in video.mkv


@dataclass(eq=False)
class Session:
    config: Config
    calibration: Calibration
    frames: list[FrameRecord] = field(default_factory=list)
    mic: np.ndarray | None = None  # float32 mono
    mic_t0: float | None = None  # monotonic time of mic[0]
    video_path: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def events(self) -> list[NoteEvent]:
        return [e for f in self.frames for e in f.events]


def save_session(session: Session, path: str | Path) -> Path:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    save_config(session.config, d / "config.toml")
    save_calibration(session.calibration, d / "calibration.toml")
    np.savez_compressed(d / "data.npz", **_encode(session.frames))
    if session.mic is not None:
        wavfile.write(str(d / "mic.wav"), session.config.mic.sample_rate, session.mic)
    meta = {
        **session.meta,
        "format": FORMAT_VERSION,
        "frames": len(session.frames),
        "events": len(session.events),
        "mic_t0": session.mic_t0,
        "video": session.video_path.name if session.video_path is not None else None,
    }
    (d / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
    return d


def load_session(path: str | Path) -> Session:
    d = Path(path)
    meta = json.loads((d / "session.json").read_text())
    if meta.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported session format: {meta.get('format')}")
    with np.load(d / "data.npz") as z:
        frames = _decode({k: z[k] for k in z.files})
    mic = None
    if (d / "mic.wav").exists():
        _, mic = wavfile.read(str(d / "mic.wav"))
        mic = np.asarray(mic, dtype=np.float32)
    video = d / meta["video"] if meta.get("video") else None
    return Session(
        config=_load_session_config(d / "config.toml"),
        calibration=load_calibration(d / "calibration.toml"),
        frames=frames,
        mic=mic,
        mic_t0=meta.get("mic_t0"),
        video_path=video,
        meta=meta,
    )


def _load_session_config(path: Path) -> Config:
    """A session's settings, without the ones this version retired (see ``_RETIRED``)."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    for section, keys in _RETIRED.items():
        if keys is None:
            data.pop(section, None)
        elif isinstance(data.get(section), dict):
            for key in keys:
                data[section].pop(key, None)
    return Config().with_overrides(data)


class TeeDetector(OnsetDetector):
    """An ``OnsetDetector`` that also copies every mic block into a ring buffer.

    ``MicStream`` runs it in the PortAudio callback; the copy is a slice assignment
    into preallocated memory, so the callback still allocates nothing per block.
    Readers on other threads call ``read_since``; a single writer and monotonically
    increasing ``written`` count make that safe without locks.
    """

    def __init__(self, cfg: MicConfig | None = None, seconds: float = 30.0) -> None:
        super().__init__(cfg)
        bs = self.cfg.block_size
        n = int(seconds * self.cfg.sample_rate) // bs * bs
        self._ring = np.zeros(n, dtype=np.float32)
        self.written = 0  # samples written since start
        self.t0: float | None = None  # monotonic time of the first sample

    def process(self, samples: np.ndarray, t: float) -> list[Onset]:
        n, size = samples.shape[0], self._ring.shape[0]
        if self.t0 is None:
            self.t0 = t
        i = self.written % size
        if i + n <= size:
            self._ring[i : i + n] = samples
        else:
            k = size - i
            self._ring[i:] = samples[:k]
            self._ring[: n - k] = samples[k:]
        self.written += n
        return super().process(samples, t)

    def read_since(self, start: int) -> tuple[np.ndarray, int]:
        """Samples from absolute index ``start`` up to now, and the new end index.

        Samples older than the ring are lost; the result then starts at the oldest kept.
        """
        end, size = self.written, self._ring.shape[0]
        start = max(start, end - size)
        idx = np.arange(start, end) % size
        return self._ring[idx].copy(), end


class Recorder:
    """Collects a live session; ``stop()`` writes it to disk.

    Video frames go to a writer thread through a small queue, so a slow disk drops
    video frames (counted in ``dropped_video``) instead of delaying play. Pass
    ``drop_video=False`` to wait instead (offline or test recording).
    """

    def __init__(
        self,
        path: str | Path,
        cfg: Config,
        calibration: Calibration,
        *,
        video: bool = True,
        mic: TeeDetector | None = None,
        drop_video: bool = True,
    ) -> None:
        self._drop_video = drop_video
        self.path = Path(path)
        self.session = Session(cfg, calibration, meta={"started": time.time()})
        self._mic = mic
        self._mic_start = mic.written if mic is not None else 0
        self._mic_chunks: list[np.ndarray] = []
        self.dropped_video = 0
        self._video_q: queue.Queue[np.ndarray | None] | None = None
        self._video_thread: threading.Thread | None = None
        self._want_video = video
        self._writer: cv2.VideoWriter | None = None

    def add(
        self,
        inp: FrameInput,
        events: list[NoteEvent],
        image: np.ndarray | None = None,
    ) -> None:
        in_video = False
        if image is not None and self._want_video:
            if self._video_q is None:
                self._start_video(image.shape[1], image.shape[0])
            if self._video_q is not None:
                try:
                    self._video_q.put(image, block=not self._drop_video)
                    in_video = True
                except queue.Full:
                    self.dropped_video += 1
        self.session.frames.append(FrameRecord(inp, tuple(events), in_video))
        if self._mic is not None:
            chunk, self._mic_start = self._mic.read_since(self._mic_start)
            if chunk.size:
                self._mic_chunks.append(chunk)

    def stop(self) -> Path:
        if self._video_q is not None:
            self._video_q.put(None)
            assert self._video_thread is not None
            self._video_thread.join()
        s = self.session
        if self._mic is not None:
            chunk, self._mic_start = self._mic.read_since(self._mic_start)
            self._mic_chunks.append(chunk)
            s.mic = np.concatenate(self._mic_chunks) if self._mic_chunks else None
            if s.mic is not None and self._mic.t0 is not None:
                first = self._mic.written - s.mic.size
                s.mic_t0 = self._mic.t0 + first / self._mic.cfg.sample_rate
        s.meta["dropped_video"] = self.dropped_video
        return save_session(s, self.path)

    def _start_video(self, width: int, height: int) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        fps = float(self.session.config.camera.fps)
        out = self.path / "video.mkv"
        self._writer = cv2.VideoWriter(
            str(out), cv2.VideoWriter_fourcc(*VIDEO_FOURCC), fps, (width, height)
        )
        if not self._writer.isOpened():
            self._want_video = False
            return
        self.session.video_path = out
        self._video_q = queue.Queue(maxsize=8)
        self._video_thread = threading.Thread(target=self._write_video, daemon=True)
        self._video_thread.start()

    def _write_video(self) -> None:
        assert self._video_q is not None and self._writer is not None
        while (img := self._video_q.get()) is not None:
            self._writer.write(img)
        self._writer.release()


# -- npz encoding ---------------------------------------------------------------------


def _encode(frames: list[FrameRecord]) -> dict[str, np.ndarray]:
    n = len(frames)
    pose = np.zeros((n, 9))
    hand_frame, hand_t, hand_lm, hand_raw = [], [], [], []
    onset_frame, onset = [], []
    ev_frame, ev_t, ev_kind, ev_string, ev_midi, ev_vel = [], [], [], [], [], []
    chords_frame, chords_json = [], []
    for i, fr in enumerate(frames):
        p = fr.inp.pose
        if fr.inp.chords is not None:
            chords_frame.append(i)
            chords_json.append(json.dumps([[n, list(v)] for n, v in fr.inp.chords]))
        pose[i] = (p.t, p.visible, *p.origin, *p.u, *p.v, p.head_len)
        for h in fr.inp.hands:
            hand_frame.append(i)
            hand_t.append(h.t)
            hand_lm.append(h.landmarks)
            hand_raw.append(h.raw)
        for o in fr.inp.onsets:
            onset_frame.append(i)
            onset.append((o.t, o.strength))
        for e in fr.events:
            ev_frame.append(i)
            ev_t.append(e.t_detect)
            ev_kind.append(_KINDS.index(e.kind))
            ev_string.append(e.string)
            ev_midi.append(-1 if e.midi is None else e.midi)
            ev_vel.append(e.velocity)
    lm_shape = (0, NUM_LANDMARKS, 2)
    return {
        "frame_t": np.array([f.inp.t for f in frames], dtype=float),
        "proc_t": np.array([f.inp.proc_t for f in frames], dtype=float),
        "in_video": np.array([f.in_video for f in frames], dtype=bool),
        "pose": pose,
        "hand_frame": np.array(hand_frame, dtype=np.int64),
        "hand_t": np.array(hand_t, dtype=float),
        "hand_lm": np.array(hand_lm, dtype=float).reshape(-1, NUM_LANDMARKS, 2)
        if hand_lm
        else np.zeros(lm_shape),
        "hand_raw": np.array(hand_raw, dtype=float).reshape(-1, NUM_LANDMARKS, 2)
        if hand_raw
        else np.zeros(lm_shape),
        "onset_frame": np.array(onset_frame, dtype=np.int64),
        "onset": np.array(onset, dtype=float).reshape(-1, 2),
        "event_frame": np.array(ev_frame, dtype=np.int64),
        "event_t": np.array(ev_t, dtype=float),
        "event_kind": np.array(ev_kind, dtype=np.int8),
        "event_string": np.array(ev_string, dtype=np.int64),
        "event_midi": np.array(ev_midi, dtype=np.int64),
        "event_velocity": np.array(ev_vel, dtype=np.int64),
        # Song-mode chord relabels: frame index and the chord set as JSON text.
        "chords_frame": np.array(chords_frame, dtype=np.int64),
        "chords_json": np.array(chords_json, dtype=str),
    }


def _decode(z: dict[str, np.ndarray]) -> list[FrameRecord]:
    n = len(z["frame_t"])
    hands: list[list[HandObs]] = [[] for _ in range(n)]
    for i, t, lm, raw in zip(
        z["hand_frame"], z["hand_t"], z["hand_lm"], z["hand_raw"], strict=True
    ):
        hands[i].append(HandObs(t=float(t), landmarks=lm.copy(), raw=raw.copy()))
    onsets: list[list[Onset]] = [[] for _ in range(n)]
    for i, (t, s) in zip(z["onset_frame"], z["onset"], strict=True):
        onsets[i].append(Onset(float(t), float(s)))
    events: list[list[NoteEvent]] = [[] for _ in range(n)]
    for i, t, k, s, m, v in zip(
        z["event_frame"],
        z["event_t"],
        z["event_kind"],
        z["event_string"],
        z["event_midi"],
        z["event_velocity"],
        strict=True,
    ):
        events[i].append(
            NoteEvent(float(t), _KINDS[int(k)], int(s), None if m < 0 else int(m), int(v))
        )
    chords: dict[int, tuple[tuple[str, tuple[int | None, ...]], ...]] = {}
    for i, text in zip(z.get("chords_frame", ()), z.get("chords_json", ()), strict=True):
        chords[int(i)] = tuple((str(name), tuple(notes)) for name, notes in json.loads(str(text)))
    frames = []
    for i in range(n):
        t, vis, ox, oy, ux, uy, vx, vy, hl = z["pose"][i]
        pose = RacketPose(
            float(t),
            bool(vis),
            Vec2(float(ox), float(oy)),
            Vec2(float(ux), float(uy)),
            Vec2(float(vx), float(vy)),
            float(hl),
        )
        inp = FrameInput(
            float(z["frame_t"][i]),
            float(z["proc_t"][i]),
            pose,
            tuple(hands[i]),
            tuple(onsets[i]),
            chords.get(i),
        )
        frames.append(FrameRecord(inp, tuple(events[i]), bool(z["in_video"][i])))
    return frames
