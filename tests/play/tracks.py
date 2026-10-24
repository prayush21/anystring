"""Track builders shared by the play tests (built on ``tests.fakes``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from racketguitar.config import Config
from racketguitar.geometry.racket import RacketFrame
from racketguitar.play.strum import StrumDetector
from racketguitar.types import HandObs, RacketPose, StringHit
from tests.fakes import hand_obs, make_pose, place_hand, strum_track, uv_to_px


def single_tip(cfg: Config) -> Config:
    """Only the index fingertip strums, so hit times follow one exact path."""
    return cfg.with_overrides({"strum": {"fingertips": [8]}})


def sweep(
    pose: RacketPose,
    *,
    v_start: float,
    v_end: float,
    duration_s: float,
    fps: float,
    u: float = 0.5,
    t0: float = 0.0,
    lead: int = 3,
    shape: str = "pick",
    jitter_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[HandObs]:
    """``lead`` still frames at ``v_start`` (to get past the re-acquire skip), then a
    straight sweep to ``v_end``. Frames are 1/fps apart throughout."""
    rng = rng or np.random.default_rng(0)
    still = strum_track(
        pose,
        v_start=v_start,
        v_end=v_start,
        u=u,
        duration_s=(lead - 1) / fps,
        fps=fps,
        t0=t0,
        shape=shape,
        jitter_px=jitter_px,
        rng=rng,
    )[:lead]
    move = strum_track(
        pose,
        v_start=v_start,
        v_end=v_end,
        u=u,
        duration_s=duration_s,
        fps=fps,
        t0=t0 + lead / fps,
        shape=shape,
        jitter_px=jitter_px,
        rng=rng,
    )
    return still + move


def tip_path(
    poses: Sequence[RacketPose],
    uvs: Sequence[tuple[float, float]],
    *,
    shape: str = "pick",
    palm_head_lengths: float = 0.27,
) -> list[HandObs]:
    """One hand per pose with the index tip at the given racket-frame point, upright to u."""
    out = []
    for pose, (u, v) in zip(poses, uvs, strict=True):
        angle = float(np.degrees(np.arctan2(pose.u.y, pose.u.x))) + 90.0
        lm = place_hand(
            shape, uv_to_px(pose, u, v), palm_px=palm_head_lengths * pose.head_len, angle_deg=angle
        )
        out.append(hand_obs(pose.t, lm))
    return out


def poses_for(frames: Sequence[HandObs], **kw) -> list[RacketPose]:
    """A still pose stamped with each frame's time."""
    return [make_pose(f.t, **kw) for f in frames]


def detect(
    cfg: Config,
    frames: Sequence[HandObs | None],
    poses: RacketPose | Sequence[RacketPose],
    *,
    low_e_neg_v: bool = True,
) -> list[StringHit]:
    det = StrumDetector(cfg.strings, cfg.strum, cfg.hands.skip_first_frames_back)
    if isinstance(poses, RacketPose):
        poses = [poses] * len(frames)
    hits: list[StringHit] = []
    for hand, pose in zip(frames, poses, strict=True):
        frame = RacketFrame.from_pose(pose) if pose.visible else None
        hits += det.update(frame, hand, low_e_neg_v)
    return hits


def at(pose: RacketPose, t: float) -> RacketPose:
    return replace(pose, t=t)
