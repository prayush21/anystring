from __future__ import annotations

import numpy as np
import pytest

from racketguitar.geometry.racket import RacketFrame
from racketguitar.play.strum import StrumDetector
from tests.fakes import expected_slots_crossed, make_pose, string_slots_v
from tests.play.tracks import at, detect, single_tip, sweep, tip_path

POSE = make_pose()


def _dirs(hits):
    return [h.direction for h in hits]


def _strings(hits):
    return [h.string for h in hits]


# Full sweeps: both fingertips (index, middle) cross every string.
@pytest.mark.parametrize("fps", [30.0, 60.0])
@pytest.mark.parametrize("duration_s", [0.5, 0.15, 0.05])  # ~1.8, 6, 18 head lengths/s
@pytest.mark.parametrize("v_start,v_end", [(-0.45, 0.45), (0.45, -0.45)])
@pytest.mark.parametrize("jitter_px", [0.0, 1.0])
def test_every_crossing_detected(cfg, rng, fps, duration_s, v_start, v_end, jitter_px):
    frames = sweep(
        POSE,
        v_start=v_start,
        v_end=v_end,
        duration_s=duration_s,
        fps=fps,
        jitter_px=jitter_px,
        rng=rng,
    )
    hits = detect(cfg, frames, POSE)
    assert _strings(hits) == expected_slots_crossed(v_start, v_end, cfg.strings)
    assert set(_dirs(hits)) == {"down" if v_end > v_start else "up"}
    ts = [h.t for h in hits]
    assert ts == sorted(ts)
    assert frames[0].t < ts[0] and ts[-1] <= frames[-1].t


@pytest.mark.parametrize("fps", [30.0, 60.0])
@pytest.mark.parametrize(
    "v_start,v_end",
    [(-0.40, 0.08), (0.08, -0.40), (-0.16, 0.20), (0.40, 0.04), (0.20, -0.08), (-0.10, 0.34)],
)
def test_partial_sweeps_single_tip(cfg, fps, v_start, v_end):
    cfg = single_tip(cfg)
    frames = sweep(POSE, v_start=v_start, v_end=v_end, duration_s=0.2, fps=fps)
    hits = detect(cfg, frames, POSE)
    assert _strings(hits) == expected_slots_crossed(v_start, v_end, cfg.strings)
    assert set(_dirs(hits)) <= {"down" if v_end > v_start else "up"}


def test_fast_sweep_between_two_frames_is_interpolated(cfg):
    """All six strings crossed in a single 1/30 s frame gap: six hits, ordered, with
    linearly interpolated timestamps."""
    cfg = single_tip(cfg)
    fps, v0, v1 = 30.0, -0.45, 0.45
    frames = sweep(POSE, v_start=v0, v_end=v1, duration_s=1 / fps, fps=fps)
    assert len(frames) == 5
    hits = detect(cfg, frames, POSE)
    assert _strings(hits) == [0, 1, 2, 3, 4, 5]
    assert _dirs(hits) == ["down"] * 6
    t0, t1 = frames[-2].t, frames[-1].t
    expected_t = t0 + (string_slots_v(cfg.strings) - v0) / (v1 - v0) * (t1 - t0)
    np.testing.assert_allclose([h.t for h in hits], expected_t, atol=1e-9)
    np.testing.assert_allclose([h.speed for h in hits], (v1 - v0) * fps, rtol=1e-9)


def test_two_fingertips_do_not_double_strike(cfg):
    """A slow sweep: the middle fingertip passes each string ~80 ms after the index,
    well past the refractory window. Still one hit per string."""
    frames = sweep(POSE, v_start=-0.45, v_end=0.45, duration_s=0.9, fps=60.0)  # 1 hl/s
    hits = detect(cfg, frames, POSE)
    assert _strings(hits) == [0, 1, 2, 3, 4, 5]
    gaps = np.diff([h.t for h in hits])
    assert np.all(gaps > 0.08)  # strings are 0.12 apart at 1 hl/s


@pytest.mark.parametrize(
    "u,expect_hits",
    [(0.149, False), (0.151, True), (0.5, True), (0.849, True), (0.851, False), (1.0, False)],
)
def test_near_misses_outside_u_range(cfg, u, expect_hits):
    cfg = single_tip(cfg)
    frames = sweep(POSE, v_start=-0.45, v_end=0.45, duration_s=0.1, fps=60.0, u=u)
    hits = detect(cfg, frames, POSE)
    assert _strings(hits) == ([0, 1, 2, 3, 4, 5] if expect_hits else [])


@pytest.mark.parametrize("u", [0.05, 0.10, 0.92, 1.1])
def test_clear_misses_with_both_fingertips(cfg, u):
    frames = sweep(POSE, v_start=-0.45, v_end=0.45, duration_s=0.1, fps=60.0, u=u)
    assert detect(cfg, frames, POSE) == []


def test_crossing_point_interpolated_along_a_diagonal_path(cfg):
    """One frame step from (0.05, -0.45) to (0.45, 0.45): the path enters the u range
    partway, so only strings whose crossing point has u >= 0.15 fire."""
    cfg = single_tip(cfg)
    a, b = np.array([0.05, -0.45]), np.array([0.45, 0.45])
    ts = np.arange(5) / 30.0
    uvs = [tuple(a)] * 4 + [tuple(b)]
    poses = [at(POSE, t) for t in ts]
    hits = detect(cfg, tip_path(poses, uvs), poses)
    slots = string_slots_v(cfg.strings)
    u_cross = a[0] + (slots - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
    expected = [i for i, u in enumerate(u_cross) if 0.15 <= u <= 0.85]
    assert 0 < len(expected) < 6
    assert _strings(hits) == expected


def test_per_string_refractory_while_other_strings_fire(cfg):
    """60 fps. string 2 down, string 3 down 17 ms later (no global cooldown), both
    back up within 40 ms (suppressed), then string 2 down again 46 ms after its last hit
    (fires) while string 3, 38 ms after its last hit, stays suppressed."""
    cfg = single_tip(cfg)
    vs = [-0.12, -0.12, -0.12, 0.0, 0.12, -0.12, 0.12]
    ts = np.arange(len(vs)) / 60.0
    poses = [at(POSE, t) for t in ts]
    hits = detect(cfg, tip_path(poses, [(0.5, v) for v in vs]), poses)
    assert [(h.string, h.direction) for h in hits] == [(2, "down"), (3, "down"), (2, "down")]
    np.testing.assert_allclose([h.t for h in hits], [0.0417, 0.0583, 0.0875], atol=1e-3)


def test_refractory_window_comes_from_config(cfg):
    cfg = single_tip(cfg.with_overrides({"strum": {"refractory_ms": 5.0}}))
    vs = [-0.12, -0.12, -0.12, 0.0, 0.12, -0.12, 0.12]
    ts = np.arange(len(vs)) / 60.0
    poses = [at(POSE, t) for t in ts]
    hits = detect(cfg, tip_path(poses, [(0.5, v) for v in vs]), poses)
    assert len(hits) == 6  # every crossing once the window is shorter than a frame


def test_min_speed(cfg):
    cfg = single_tip(cfg)
    slow = sweep(POSE, v_start=-0.45, v_end=0.45, duration_s=0.9 / 0.7, fps=60.0)  # 0.7 hl/s
    fast = sweep(POSE, v_start=-0.45, v_end=0.45, duration_s=0.9 / 0.9, fps=60.0)  # 0.9 hl/s
    assert detect(cfg, slow, POSE) == []
    hits = detect(cfg, fast, POSE)
    assert len(hits) == 6
    assert all(h.speed == pytest.approx(0.9, rel=1e-6) for h in hits)


def test_no_hits_while_pose_not_visible(cfg):
    frames = sweep(POSE, v_start=-0.45, v_end=0.45, duration_s=0.1, fps=60.0)
    hidden = [make_pose(f.t, visible=False) for f in frames]
    assert detect(cfg, frames, hidden) == []


def test_pose_dropout_breaks_the_path(cfg):
    """The racket vanishes for the frame in which the strings are crossed: no hits."""
    cfg = single_tip(cfg)
    vs = [-0.45, -0.45, -0.45, -0.45, 0.45, 0.45]
    ts = np.arange(len(vs)) / 60.0
    poses = [at(POSE, t) for t in ts]
    poses[4] = make_pose(ts[4], visible=False)
    assert detect(cfg, tip_path(poses, [(0.5, v) for v in vs]), poses) == []


@pytest.mark.parametrize("skip", [0, 1, 2])
def test_skip_first_frames_back(cfg, skip):
    """The hand is lost, then re-acquired at v = -0.45 and sweeps one string per frame.
    The first ``skip`` frames back are not used, so the crossings they bound are lost."""
    cfg = single_tip(cfg.with_overrides({"hands": {"skip_first_frames_back": skip}}))
    slots = string_slots_v(cfg.strings)
    vs = [-0.45, *(slots + 0.06)]  # each step crosses exactly one string
    ts = np.arange(len(vs) + 2) / 60.0
    poses = [at(POSE, t) for t in ts]
    hands = tip_path(poses[2:], [(0.5, v) for v in vs])
    frames = [None, None, *hands]
    hits = detect(cfg, frames, poses)
    # Crossing k happens between hand frames k and k+1; frames 0..skip-1 are dropped,
    # so the first usable segment starts at hand frame ``skip``.
    assert _strings(hits) == list(range(skip, 6))


def test_hand_lost_mid_sweep_is_not_bridged(cfg):
    cfg = single_tip(cfg)
    vs = [-0.45, -0.45, -0.45, -0.20, None, 0.20, 0.45, 0.45]
    ts = np.arange(len(vs)) / 60.0
    poses = [at(POSE, t) for t in ts]
    frames = [
        None if v is None else tip_path([p], [(0.5, v)])[0] for p, v in zip(poses, vs, strict=True)
    ]
    hits = detect(cfg, frames, poses)
    # Before the loss: slot 0 (-0.30). After: frame at 0.20 is skipped, 0.45 is the
    # first sample, so nothing more. The 4 strings between -0.20 and 0.20 never fire.
    assert _strings(hits) == [0]


def test_tracking_jump_is_not_a_strum(cfg):
    """A sample that teleports across the head (e.g. swapped hand identity) is treated
    as a re-acquire, not a 6-string sweep."""
    cfg = single_tip(cfg)
    vs = [-0.45, -0.45, -0.45, 0.45, 0.45, 0.45]
    ts = np.arange(len(vs)) / 240.0  # 0.9 hl in 4 ms = 216 hl/s
    poses = [at(POSE, t) for t in ts]
    assert detect(cfg, tip_path(poses, [(0.5, v) for v in vs]), poses) == []


def test_moving_rotating_racket(cfg, rng):
    """Racket translates, rotates 40 degrees and shrinks while the fingertip sweeps in
    its frame: every string, in order, in each frame's own coordinates."""
    n = 20
    ts = np.arange(n) / 30.0
    poses = [
        make_pose(
            t,
            origin=(600.0 + 15 * i, 480.0 - 8 * i),
            angle_deg=-90.0 + 2.0 * i,
            head_len=300.0 - 4 * i,
        )
        for i, t in enumerate(ts)
    ]
    vs = np.r_[[-0.45] * 3, np.linspace(-0.45, 0.45, n - 3)]
    frames = tip_path(poses, [(0.5, v) for v in vs])
    hits = detect(cfg, frames, poses)
    assert _strings(hits) == [0, 1, 2, 3, 4, 5]
    assert _dirs(hits) == ["down"] * 6
    # And back up across the moving racket.
    hits = detect(cfg, tip_path(poses, [(0.5, v) for v in vs[::-1]]), poses)
    assert _strings(hits) == [5, 4, 3, 2, 1, 0]
    assert _dirs(hits) == ["up"] * 6


def test_hand_moving_with_racket_is_silent(cfg):
    """Fingertip fixed in racket coordinates while the racket swings fast in pixels."""
    n = 15
    ts = np.arange(n) / 30.0
    poses = [
        make_pose(t, origin=(400.0 + 40 * i, 500.0), angle_deg=-90.0 + 5 * i)
        for i, t in enumerate(ts)
    ]
    assert detect(cfg, tip_path(poses, [(0.5, 0.0)] * n), poses) == []


def test_racket_moving_under_still_hand_strums(cfg):
    """Fingertip still in pixels, racket slides along +v under it: in the racket frame
    the tip moves toward -v, so the strings fire high E to low E as up-strums."""
    cfg = single_tip(cfg)
    n = 12
    ts = np.arange(n) / 60.0
    shift = np.r_[[0.0] * 3, np.linspace(0.0, 0.9, n - 3)] * 300.0  # pixels along v=(1,0)
    poses = [make_pose(t, origin=(640.0 + s, 500.0)) for t, s in zip(ts, shift, strict=True)]
    tip_px_frames = tip_path([poses[0]] * n, [(0.5, 0.45)] * n)
    frames = [
        type(f)(t=float(t), landmarks=f.landmarks, raw=f.raw)
        for f, t in zip(tip_px_frames, ts, strict=True)
    ]
    hits = detect(cfg, frames, poses)
    assert _strings(hits) == [5, 4, 3, 2, 1, 0]
    assert _dirs(hits) == ["up"] * 6


def test_pos_v_orientation_reverses_strings_and_directions(cfg):
    frames = sweep(POSE, v_start=-0.45, v_end=0.45, duration_s=0.1, fps=60.0)
    hits = detect(cfg, frames, POSE, low_e_neg_v=False)
    assert _strings(hits) == [5, 4, 3, 2, 1, 0]
    assert _dirs(hits) == ["up"] * 6


def test_alternating_strums_8_per_second(cfg, rng):
    """Down/up at 8 strums per second for 2 s at 30 fps with jitter: no drops."""
    fps, n_strums = 30.0, 16
    period = 1.0 / 8
    ts = np.arange(round(n_strums * period * fps) + 4) / fps
    phase = np.clip((ts - 3 / fps) / period, 0, None)
    # Triangle wave between -0.6 and +0.6 (turning well clear of the outer strings, so
    # the refractory window never spans a turn), one sweep per period, starting down.
    tri = np.abs((phase % 2) - 1)  # 1 -> 0 -> 1
    vs = -0.6 + 1.2 * (1 - tri)
    poses = [at(POSE, t) for t in ts]
    frames = tip_path(poses, [(0.5, v) for v in vs])
    for f in frames:
        f.landmarks[:] += rng.normal(0, 1.0, f.landmarks.shape)
    hits = detect(cfg, frames, poses)
    assert len(hits) == 6 * n_strums
    for k in range(n_strums):
        block = hits[6 * k : 6 * k + 6]
        down = k % 2 == 0
        assert _strings(block) == ([0, 1, 2, 3, 4, 5] if down else [5, 4, 3, 2, 1, 0])
        assert set(_dirs(block)) == {"down" if down else "up"}


def test_reset_forgets_the_hand(cfg):
    cfg = single_tip(cfg)
    det = StrumDetector(cfg.strings, cfg.strum, cfg.hands.skip_first_frames_back)
    frame = RacketFrame.from_pose(POSE)
    frames = sweep(POSE, v_start=-0.45, v_end=0.45, duration_s=1 / 60, fps=60.0)
    for f in frames[:-1]:
        det.update(frame, f, True)
    det.reset()
    assert det.update(frame, frames[-1], True) == []
    assert det.tips_uv is not None


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_slow_jittery_sweeps_are_not_lost(cfg, fps):
    """1 head length/s (just above min_speed) with 1 px landmark jitter, 30 seeds: the
    windowed speed keeps every crossing above threshold."""
    for seed in range(30):
        rng = np.random.default_rng(seed)
        frames = sweep(
            POSE, v_start=-0.45, v_end=0.45, duration_s=0.9, fps=fps, jitter_px=1.0, rng=rng
        )
        assert _strings(detect(cfg, frames, POSE)) == [0, 1, 2, 3, 4, 5], seed


def test_speed_restarts_at_a_reversal(cfg):
    """A sharp turn just past the top string: the return crossing's speed comes from the
    return stroke only, not from a window spanning the turn."""
    cfg = single_tip(cfg.with_overrides({"strum": {"refractory_ms": 5.0}}))
    vs = [-0.45, -0.45, -0.45, -0.20, 0.05, 0.40, 0.25]
    ts = np.arange(len(vs)) / 60.0
    poses = [at(POSE, t) for t in ts]
    hits = detect(cfg, tip_path(poses, [(0.5, v) for v in vs]), poses)
    assert [(h.string, h.direction) for h in hits][-2:] == [(5, "down"), (5, "up")]
    assert hits[-1].speed == pytest.approx(0.15 * 60)
    # The down-strum's speed averages its whole run within the window.
    assert hits[-2].speed == pytest.approx((0.40 - -0.45) / (3 / 60))


def test_full_down_up_with_sharp_turns(cfg):
    up = -0.455 + 0.13 * np.arange(8)  # 3.9 hl/s at 30 fps, turning 0.155 past the end
    vs = np.r_[[-0.455] * 3, up, up[::-1][1:]]
    ts = np.arange(len(vs)) / 30.0
    poses = [at(POSE, t) for t in ts]
    hits = detect(cfg, tip_path(poses, [(0.5, v) for v in vs]), poses)
    assert _strings(hits) == [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]
    assert _dirs(hits) == ["down"] * 6 + ["up"] * 6
