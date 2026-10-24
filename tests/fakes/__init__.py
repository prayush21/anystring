"""Fake data for headless tests. Test-only: production code under racketguitar/ never
imports from here.

    from tests.fakes import make_pose, strum_track, brush_strokes, render_marker_scene
"""

from .audio import (
    SR,
    blocks,
    brush_strokes,
    mix,
    read_wav,
    room_noise,
    silence,
    speaker_bleed,
    write_wav,
)
from .gates import FakeClock, ScriptedGate
from .hands import fretting_hand, hand_obs, hand_template, place_hand, strum_track
from .racket import expected_slots_crossed, make_pose, px_to_uv, string_slots_v, uv_to_px
from .scenes import MarkerScene, render_marker_scene

__all__ = [
    "SR",
    "FakeClock",
    "MarkerScene",
    "ScriptedGate",
    "blocks",
    "brush_strokes",
    "expected_slots_crossed",
    "fretting_hand",
    "hand_obs",
    "hand_template",
    "make_pose",
    "mix",
    "place_hand",
    "px_to_uv",
    "read_wav",
    "render_marker_scene",
    "room_noise",
    "silence",
    "speaker_bleed",
    "string_slots_v",
    "strum_track",
    "uv_to_px",
    "write_wav",
]
