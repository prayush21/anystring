"""The grip threshold sweep on a recorded synthetic take: pick-grip downs, relaxed returns."""

from __future__ import annotations

import pytest

from anystring.config import Calibration
from anystring.pipeline import FrameInput
from anystring.replay.record import Recorder
from tests.fakes import hand_obs, make_pose, place_hand, uv_to_px
from tests.integration.test_strum_gate import performance
from tools.grip_sweep import Setting, evaluate, load_take, main, sweep, take_parts


def record_relaxed_take(cfg, path, strokes: int = 3):
    rec = Recorder(path, cfg, Calibration(), video=False, mic=None)
    t_off = 0.0
    for _ in range(strokes):
        frames = performance(0.6, 2)
        for t, v, shape in frames:
            t += t_off
            pose = make_pose(t)
            lm = place_hand(shape, uv_to_px(pose, 0.5, v), palm_px=0.27 * pose.head_len)
            rec.add(FrameInput(t, t, pose, (hand_obs(t, lm),), ()), [], None)
        t_off += frames[-1][0] - frames[0][0] + 0.5
    rec.stop()


def test_sweep_separates_blocking_from_leaking_settings(tmp_path, cfg):
    record_relaxed_take(cfg, tmp_path / "relaxed")
    take = load_take(tmp_path / "relaxed", "relaxed")
    assert sum(d == "down" for _, d in take.hits) == 18
    assert sum(d == "up" for _, d in take.hits) == 18

    strict = evaluate([take], Setting(0.25, 0.5, 2, 1))
    assert (strict.missed, strict.extra) == (0, 0)
    old = evaluate([take], Setting(0.25, 0.75, 2, 2))
    assert old.missed == 0 and old.extra == 18  # every relaxed return played

    best = sweep([take])[0]
    assert best.errors == 0


def test_cli_lists_settings(tmp_path, cfg, capsys):
    record_relaxed_take(cfg, tmp_path / "relaxed", strokes=1)
    assert main(["--relaxed", str(tmp_path / "relaxed"), "--top", "3", "--hold", "2"]) == 0
    out = capsys.readouterr().out
    assert "relaxed down" in out and "(default)" in out and "(old thresholds)" in out


def test_a_base_name_includes_its_parts_and_skips_empty_ones(tmp_path, cfg, capsys):
    """The app saves the lead-up to locking on as the base folder and the playing as
    -part2; naming the base must score the playing."""
    lead_up = Recorder(tmp_path / "relaxed", cfg, Calibration(), video=False, mic=None)
    lead_up.add(FrameInput(99.0, 99.0, make_pose(99.0), (), ()), [], None)
    lead_up.stop()
    record_relaxed_take(cfg, tmp_path / "relaxed-part2", strokes=1)
    assert take_parts(tmp_path / "relaxed") == [tmp_path / "relaxed", tmp_path / "relaxed-part2"]
    assert main(["--relaxed", str(tmp_path / "relaxed"), "--hold", "2"]) == 0
    out = capsys.readouterr().out
    assert "relaxed-part2: 12 string crossings" in out and "relaxed up" in out


def test_takes_without_crossings_are_an_error(tmp_path, cfg):
    empty = Recorder(tmp_path / "closed", cfg, Calibration(), video=False, mic=None)
    empty.add(FrameInput(99.0, 99.0, make_pose(99.0), (), ()), [], None)
    empty.stop()
    with pytest.raises(SystemExit) as exc:
        main(["--closed", str(tmp_path / "closed")])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        main(["--closed", str(tmp_path / "missing")])
