"""A scripted performance for end-to-end tests: C, G, Am, F strummed down and up, an
open-hand lift-away that must stay silent, and a resting palm that mutes.

The racket drifts and rotates slowly while playing, as a held racket does. Each frame
gives the true pose and both hands placed relative to it. The brush times returned by
``frames`` are when any strumming fingertip crosses a string segment during an armed
(pick-grip) sweep: the moments a real string would sound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from racketguitar.config import Config
from racketguitar.types import HandObs, RacketPose
from tests.fakes import fretting_hand, make_pose, place_hand, px_to_uv, string_slots_v, uv_to_px
from tests.fakes.hands import INDEX_TIP, hand_obs

T0 = 100.0
REST_V = -0.45  # strumming fingertip parked beyond the low-E side


@dataclass(frozen=True)
class Segment:
    dur: float
    v0: float
    v1: float
    shape: str  # "pick", "open" or "fist"
    zone: int | None  # handle section the fretting hand grips, None = no fretting hand
    u: float = 0.5


def script() -> list[Segment]:
    segs: list[Segment] = []
    for zone in range(4):  # C, G, Am, F
        segs += [
            Segment(0.12, REST_V, REST_V, "pick", zone),  # settle grip and chord
            Segment(0.10, REST_V, -REST_V, "pick", zone),  # down
            Segment(0.06, -REST_V, -REST_V, "pick", zone),
            Segment(0.10, -REST_V, REST_V, "pick", zone),  # up
        ]
    segs += [
        Segment(0.10, REST_V, REST_V, "open", 3),
        Segment(0.12, REST_V, -REST_V, "open", 3),  # open hand across strings: silent
        Segment(0.30, 0.0, 0.0, "open", 3, u=0.78),  # palm resting on the head: mute
    ]
    return segs


@dataclass(frozen=True, eq=False)
class SongFrame:
    t: float
    pose: RacketPose
    hands: tuple[HandObs, ...]
    scene: dict  # tests.fakes.render_racket kwargs giving this pose
    armed: bool  # strumming hand in a pick grip


def pose_at(t: float) -> tuple[RacketPose, dict]:
    """Slow drift: a little translation and rotation, constant size."""
    k = t - T0
    angle = -90.0 + 6.0 * math.sin(1.3 * k)
    centre = (640.0 + 25.0 * math.sin(0.9 * k), 380.0 + 10.0 * math.cos(1.1 * k))
    head_len = 300.0
    a = math.radians(angle)
    # The scene is placed by the head centre; the pose origin is the throat.
    origin = (centre[0] - 0.5 * head_len * math.cos(a), centre[1] - 0.5 * head_len * math.sin(a))
    pose = make_pose(t, origin=origin, angle_deg=angle, head_len=head_len)
    return pose, {"center_px": centre, "angle_deg": angle, "head_len_px": head_len}


def frames(cfg: Config, fps: float = 60.0) -> tuple[list[SongFrame], list[float]]:
    zones = len(cfg.chords.chords)
    out: list[SongFrame] = []
    t_seg = T0
    k = 0
    for seg in script():
        end = t_seg + seg.dur
        while T0 + k / fps < end - 1e-9:
            t = T0 + k / fps
            frac = (t - t_seg) / seg.dur
            v = seg.v0 + frac * (seg.v1 - seg.v0)
            pose, scene = pose_at(t)
            angle = math.degrees(math.atan2(pose.u.y, pose.u.x)) + 90.0
            tip = uv_to_px(pose, seg.u, v)
            strum = hand_obs(
                t,
                place_hand(
                    seg.shape, tip, anchor=INDEX_TIP, palm_px=0.27 * pose.head_len, angle_deg=angle
                ),
            )
            hands = [strum]
            if seg.zone is not None:
                ht = (seg.zone + 0.5) / zones
                hands.append(
                    fretting_hand(
                        pose,
                        t,
                        handle_t=ht,
                        handle_u_near=cfg.chords.handle_u_near,
                        handle_u_far=cfg.chords.handle_u_far,
                    )
                )
            out.append(SongFrame(t, pose, tuple(hands), scene, seg.shape == "pick"))
            k += 1
        t_seg = end
    return out, _contacts(out, cfg)


def _contacts(song: list[SongFrame], cfg: Config) -> list[float]:
    slots = string_slots_v(cfg.strings)
    lo, hi = cfg.strings.u_min, cfg.strings.u_max
    times = []
    for a, b in zip(song, song[1:], strict=False):
        if not (a.armed and b.armed):
            continue
        for tip in cfg.strum.fingertips:
            ua, va = px_to_uv(a.pose, a.hands[0].landmarks[tip])
            ub, vb = px_to_uv(b.pose, b.hands[0].landmarks[tip])
            for sv in slots:
                if (va - sv) * (vb - sv) < 0:
                    f = (sv - va) / (vb - va)
                    if lo <= ua + f * (ub - ua) <= hi:
                        times.append(a.t + f * (b.t - a.t))
    return sorted(times)


def chord_of(midi: int, cfg: Config) -> set[str]:
    return {name for name, notes in cfg.chords.chords if midi in notes}


def expected_note_count(cfg: Config) -> int:
    """Down + up per chord, each over the chord's played strings."""
    return sum(2 * sum(n is not None for n in notes) for _, notes in cfg.chords.chords)
