from __future__ import annotations

import numpy as np
import pytest

from anystring.config import ChordConfig, StringsConfig
from anystring.geometry.racket import (
    HandleAxis,
    RacketFrame,
    StringLayout,
    StringOrientation,
    in_head,
)
from tests.fakes import make_pose, px_to_uv, string_slots_v, uv_to_px


@pytest.mark.parametrize("angle", [-90.0, 0.0, 37.0, 180.0, -143.0])
@pytest.mark.parametrize("head_len", [120.0, 300.0])
def test_frame_round_trip_matches_oracle(angle, head_len, rng):
    pose = make_pose(origin=(500.0, 300.0), angle_deg=angle, head_len=head_len)
    frame = RacketFrame.from_pose(pose)
    uv = rng.uniform(-1.2, 1.2, (50, 2))
    px = frame.to_px(uv)
    np.testing.assert_allclose(px, uv_to_px(pose, uv[:, 0], uv[:, 1]), atol=1e-9)
    np.testing.assert_allclose(frame.to_uv(px), uv, atol=1e-12)
    np.testing.assert_allclose(frame.to_uv(px), px_to_uv(pose, px), atol=1e-12)
    np.testing.assert_allclose(frame.v, pose.v, atol=1e-12)


def test_string_segments(cfg):
    layout = StringLayout.from_config(cfg.strings)
    np.testing.assert_allclose(layout.slot_v, string_slots_v(cfg.strings))
    pose = make_pose(angle_deg=20.0)
    frame = RacketFrame.from_pose(pose)
    seg = layout.segments_px(frame, low_e_neg_v=True)
    assert seg.shape == (6, 2, 2)
    for s, v in enumerate(string_slots_v(cfg.strings)):
        np.testing.assert_allclose(seg[s, 0], uv_to_px(pose, 0.15, v))
        np.testing.assert_allclose(seg[s, 1], uv_to_px(pose, 0.85, v))
    flipped = layout.segments_px(frame, low_e_neg_v=False)
    np.testing.assert_allclose(flipped, seg[::-1])
    assert [layout.string_of_slot(i, False) for i in range(6)] == [5, 4, 3, 2, 1, 0]


def test_handle_axis():
    handle = HandleAxis.from_config(ChordConfig())
    assert handle.t_of(np.array([-0.15, 0.0])) == pytest.approx(0.0)
    assert handle.t_of(np.array([-1.0, 0.2])) == pytest.approx(1.0)
    assert handle.t_of(np.array([-0.575, 0.0])) == pytest.approx(0.5)
    assert handle.distance(np.array([-0.5, 0.1])) == pytest.approx(0.1)
    assert handle.distance(np.array([0.15, 0.0])) == pytest.approx(0.3)  # past the near end
    pose = make_pose(angle_deg=-30.0)
    sec = handle.sections_px(RacketFrame.from_pose(pose), 4)
    np.testing.assert_allclose(sec[0, 0], uv_to_px(pose, -0.15, 0.0))
    np.testing.assert_allclose(sec[3, 1], uv_to_px(pose, -1.0, 0.0))
    np.testing.assert_allclose(sec[1, 0], sec[0, 1])


def test_in_head():
    assert in_head(np.array([0.5, 0.0]))
    assert in_head(np.array([0.5, 0.39]))
    assert not in_head(np.array([0.5, 0.41]))
    assert not in_head(np.array([-0.05, 0.0]))
    assert not in_head(np.array([1.05, 0.0]))


def _orient(side, angle):
    o = StringOrientation(side)
    return o.update(RacketFrame.from_pose(make_pose(angle_deg=angle)))


def test_fixed_orientation():
    assert _orient("neg_v", 0.0) is True
    assert _orient("neg_v", 180.0) is True
    assert _orient("pos_v", 0.0) is False


def test_auto_orientation_puts_low_e_on_top():
    # u to the right (handle left): v points down, so the neg-v side is on top.
    assert _orient("auto", 0.0) is True
    # u to the left (handle right): v points up, so low E is on the pos-v side.
    assert _orient("auto", 180.0) is False
    # Upright racket: v horizontal, tie -> neg_v.
    assert _orient("auto", -90.0) is True


def test_auto_orientation_hysteresis():
    o = StringOrientation("auto")
    margin_deg = np.degrees(np.arcsin(StringsConfig().auto_side_margin))
    # Start upright, rock either way inside the margin: never flips.
    sides = [
        o.update(RacketFrame.from_pose(make_pose(angle_deg=a)))
        for a in (-90.0, -90 - margin_deg + 2, -90 + margin_deg - 2, -90 - margin_deg + 1)
    ]
    assert sides == [True] * 4
    # Past the margin toward a left-pointing u (v.y < 0): flips to pos_v ...
    assert o.update(RacketFrame.from_pose(make_pose(angle_deg=-90 - margin_deg - 2))) is False
    # ... and stays there while back inside the margin.
    assert o.update(RacketFrame.from_pose(make_pose(angle_deg=-90.0))) is False
    assert o.update(RacketFrame.from_pose(make_pose(angle_deg=-90 + margin_deg - 1))) is False
    assert o.update(RacketFrame.from_pose(make_pose(angle_deg=-90 + margin_deg + 2))) is True


def test_layout_rejects_mismatched_tuning():
    with pytest.raises(ValueError):
        StringLayout.from_config(StringsConfig(count=5))
