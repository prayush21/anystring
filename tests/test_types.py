import dataclasses

import numpy as np
import pytest

from anystring.types import (
    HandObs,
    NoteEvent,
    Onset,
    RacketPose,
    StringHit,
    StrumGate,
    Vec2,
)
from tests.fakes import ScriptedGate


def test_contracts_are_frozen():
    pose = RacketPose(0.0, True, Vec2(1, 2), Vec2(1, 0), Vec2(0, 1), 100.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pose.visible = False  # type: ignore[misc]
    for cls in (RacketPose, HandObs, Onset, StringHit, NoteEvent):
        assert cls.__dataclass_params__.frozen


def test_contract_fields_match_pid():
    names = {
        cls.__name__: [f.name for f in dataclasses.fields(cls)]
        for cls in (RacketPose, HandObs, Onset, StringHit, NoteEvent)
    }
    assert names == {
        "RacketPose": ["t", "visible", "origin", "u", "v", "head_len"],
        "HandObs": ["t", "landmarks", "raw"],
        "Onset": ["t", "strength"],
        "StringHit": ["t", "string", "direction", "speed"],
        "NoteEvent": ["t_detect", "kind", "string", "midi", "velocity"],
    }


def test_vec2_is_array_friendly():
    v = Vec2(3.0, 4.0)
    assert np.hypot(*v) == 5.0
    assert np.asarray(v).tolist() == [3.0, 4.0]


def test_hand_obs_holds_arrays():
    lm = np.zeros((21, 2))
    obs = HandObs(t=1.0, landmarks=lm, raw=lm)
    assert obs.landmarks.shape == (21, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind="on", string=0, midi=40, velocity=100),
        dict(kind="damp", string=5, midi=None, velocity=1),
        dict(kind="all_off", string=-1, midi=None, velocity=1),
    ],
)
def test_note_event_valid(kwargs):
    NoteEvent(t_detect=0.0, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind="on", string=0, midi=None, velocity=100),
        dict(kind="on", string=6, midi=40, velocity=100),
        dict(kind="on", string=0, midi=40, velocity=0),
        dict(kind="on", string=0, midi=40, velocity=128),
        dict(kind="damp", string=0, midi=40, velocity=1),
        dict(kind="all_off", string=0, midi=None, velocity=1),
        dict(kind="bend", string=0, midi=40, velocity=1),
    ],
)
def test_note_event_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        NoteEvent(t_detect=0.0, **kwargs)


def test_string_hit_validation():
    StringHit(t=0.0, string=3, direction="up", speed=2.0)
    with pytest.raises(ValueError):
        StringHit(t=0.0, string=-1, direction="up", speed=2.0)
    with pytest.raises(ValueError):
        StringHit(t=0.0, string=0, direction="sideways", speed=2.0)  # type: ignore[arg-type]


def test_scripted_gate_satisfies_protocol():
    gate = ScriptedGate([(1.0, 2.0)])
    assert isinstance(gate, StrumGate)
    assert not gate.allows(0.5) and gate.allows(1.5) and not gate.allows(2.0)
