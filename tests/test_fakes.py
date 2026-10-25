"""Sanity checks on the fake data, so work packages can trust it as ground truth."""

import numpy as np
import pytest

from tests.fakes import (
    SR,
    ScriptedGate,
    blocks,
    brush_strokes,
    expected_slots_crossed,
    fretting_hand,
    hand_template,
    make_pose,
    mix,
    px_to_uv,
    read_wav,
    room_noise,
    speaker_bleed,
    strum_track,
    uv_to_px,
    write_wav,
)
from tests.fakes.hands import INDEX_PIP, INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST


@pytest.mark.parametrize("angle", [-90.0, 0.0, 37.0, 180.0])
def test_pose_convention_and_round_trip(angle):
    pose = make_pose(angle_deg=angle, head_len=250.0)
    assert np.isclose(np.hypot(*pose.u), 1.0)
    assert np.allclose(pose.v, (-pose.u.y, pose.u.x))
    assert np.allclose(
        uv_to_px(pose, 1.0, 0.0), np.asarray(pose.origin) + np.asarray(pose.u) * 250.0
    )
    pts = np.array([[0.3, -0.2], [0.85, 0.3], [-0.5, 0.0]])
    assert np.allclose(px_to_uv(pose, uv_to_px(pose, pts[:, 0], pts[:, 1])), pts)


def _palm(lm):
    return np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST])


def test_hand_shapes_are_distinguishable():
    pick, open_ = hand_template("pick"), hand_template("open")
    assert np.isclose(_palm(pick), 1.0) and np.isclose(_palm(open_), 1.0)
    # Pick grip: thumb tip on the side of the index finger; open: far apart.
    assert np.linalg.norm(pick[THUMB_TIP] - pick[INDEX_PIP]) < 0.2
    assert np.linalg.norm(open_[THUMB_TIP] - open_[INDEX_PIP]) > 0.6
    # Open fingers extend well past the knuckles; curled ones do not.
    for tip in (8, 12, 16, 20):
        assert np.linalg.norm(open_[tip] - open_[WRIST]) > 1.4
        assert np.linalg.norm(pick[tip] - pick[WRIST]) < 1.1


def test_strum_track_follows_racket_frame():
    pose = make_pose(angle_deg=-70.0)
    frames = strum_track(pose, v_start=-0.4, v_end=0.4, u=0.5, duration_s=0.1, fps=60)
    assert len(frames) == 7
    uv = np.array([px_to_uv(pose, f.landmarks[INDEX_TIP]) for f in frames])
    assert np.allclose(uv[:, 0], 0.5)
    assert np.allclose(uv[[0, -1], 1], [-0.4, 0.4])
    assert np.all(np.diff([f.t for f in frames]) > 0)
    assert expected_slots_crossed(-0.4, 0.4) == [0, 1, 2, 3, 4, 5]
    assert expected_slots_crossed(0.4, -0.13) == [5, 4, 3, 2]
    assert expected_slots_crossed(0.35, 0.5) == []


def test_fretting_hand_knuckle_on_handle():
    pose = make_pose()
    hand = fretting_hand(pose, 0.0, handle_t=0.5)
    u, v = px_to_uv(pose, hand.landmarks[MIDDLE_MCP])
    assert np.isclose(u, -0.575) and np.isclose(v, 0.0)


def test_audio_fakes():
    sig, onsets = brush_strokes([0.1, 0.3], 0.5)
    noise = room_noise(0.5, rms=0.003)
    bleed, _ = speaker_bleed([(0.2, 110.0)], 0.5)
    x = mix(sig, noise, bleed)
    assert len(x) == round(0.5 * SR) and x.dtype == np.float32
    assert np.isclose(np.sqrt(np.mean(noise**2)), 0.003, rtol=1e-3)
    # Bursts are loud right after each onset and quiet just before it.
    for t in onsets:
        i = round(t * SR)
        assert np.abs(sig[i : i + 480]).max() > 0.1
        assert np.abs(sig[i - 480 : i]).max() < 1e-6
    bs = list(blocks(x, 64))
    assert all(b.shape == (64,) for b in bs) and len(bs) == -(-len(x) // 64)


def test_bleed_energy_sits_below_brush_band():
    bleed, _ = speaker_bleed([(0.0, 196.0)], 0.5)
    spec = np.abs(np.fft.rfft(bleed)) ** 2
    f = np.fft.rfftfreq(len(bleed), 1 / SR)
    assert spec[(f >= 2_000) & (f <= 8_000)].sum() < 0.01 * spec.sum()


def test_wav_round_trip(tmp_path):
    x, _ = brush_strokes([0.05], 0.1)
    write_wav(tmp_path / "b.wav", x)
    y, sr = read_wav(tmp_path / "b.wav")
    assert sr == SR and np.allclose(x, y)


def test_scripted_gate_records_calls():
    g = ScriptedGate.always()
    g.observe_hand(1.0, None)
    g.note_emitted(1.0)
    assert g.allows(1.0) and g.grip == "closed"
    assert ScriptedGate.never().grip == "unknown"
