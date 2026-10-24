import pytest

from racketguitar.config import (
    DEFAULT_CHORDS,
    STANDARD_TUNING,
    Calibration,
    Config,
    home_dir,
    load_calibration,
    load_config,
    save_calibration,
    save_config,
)


def test_defaults_match_pid():
    c = Config()
    assert c.strings.tuning == STANDARD_TUNING == (40, 45, 50, 55, 59, 64)
    assert (c.strings.u_min, c.strings.u_max) == (0.15, 0.85)
    assert (c.strings.v_min, c.strings.v_max) == (-0.30, 0.30)
    assert c.strum.refractory_ms == 40.0
    assert c.gate.pick_grip is True and c.gate.mic is False
    assert c.gate.onset_lookback_ms == 80.0 and c.gate.onset_hold_ms == 150.0
    assert c.velocity.gamma == 1.6 and c.velocity.min_velocity == 20
    assert [name for name, _ in c.chords.chords] == ["C", "G", "Am", "F"]
    assert c.mic.block_size == 64 and c.audio.block_size == 128


def test_chord_voicings_have_six_strings():
    for _, voicing in DEFAULT_CHORDS:
        assert len(voicing) == 6
        assert all(n is None or 36 <= n <= 84 for n in voicing)


def test_overrides_are_partial():
    c = Config().with_overrides({"gate": {"mic": True}, "strings": {"v_min": -0.25}})
    assert c.gate.mic is True and c.gate.pick_grip is True
    assert c.strings.v_min == -0.25 and c.strings.v_max == 0.30


@pytest.mark.parametrize("bad", [{"gaet": {}}, {"gate": {"mik": True}}, {"gate": 3}])
def test_unknown_keys_rejected(bad):
    with pytest.raises(ValueError):
        Config().with_overrides(bad)


def test_config_round_trip(tmp_path):
    c = Config().with_overrides({"audio": {"backend": "fluid"}, "ui": {"mirror": False}})
    path = tmp_path / "config.toml"
    save_config(c, path)
    assert load_config(path) == c


def test_missing_default_config_gives_defaults():
    assert not (home_dir() / "config.toml").exists()
    assert load_config() == Config()


def test_explicit_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_calibration_round_trip():
    assert load_calibration() == Calibration()
    cal = Calibration(grip_closed_score=0.4, grip_open_score=1.3, mic_noise_floor=1e-4)
    save_calibration(cal)
    assert (home_dir() / "calibration.toml").exists()
    assert load_calibration() == cal
