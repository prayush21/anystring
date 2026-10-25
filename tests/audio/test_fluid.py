"""FluidSynth backend: event mapping against a fake module; real playback only when
pyfluidsynth and a soundfont are available (skipped otherwise)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from anystring.audio import fluid
from anystring.audio.fluid import FluidSynthBackend, fluidsynth_available
from anystring.audio.synth import Synth, make_synth
from anystring.config import AudioConfig
from tests.audio.measure import all_off, damp, on


class FakeFluidSynth:
    """Records calls; renders a constant so the sample conversion can be checked."""

    def __init__(self, gain=0.2, samplerate=44100.0):
        self.gain, self.samplerate = gain, samplerate
        self.calls: list[tuple] = []

    def sfload(self, path):
        self.calls.append(("sfload", path))
        return -1 if path == "missing.sf2" else 1

    def program_select(self, ch, sfid, bank, preset):
        self.calls.append(("program", ch, sfid, bank, preset))

    def noteon(self, ch, key, vel):
        self.calls.append(("on", ch, key, vel))

    def noteoff(self, ch, key):
        self.calls.append(("off", ch, key))

    def cc(self, ch, ctrl, val):
        self.calls.append(("cc", ch, ctrl, val))

    def get_samples(self, n):
        s = np.empty(2 * n, dtype=np.int16)
        s[0::2], s[1::2] = 16384, -8192  # left 0.5, right -0.25
        return s

    def delete(self):
        self.calls.append(("delete",))


@pytest.fixture
def fake_module(monkeypatch):
    mod = types.ModuleType("fluidsynth")
    mod.Synth = FakeFluidSynth
    monkeypatch.setitem(sys.modules, "fluidsynth", mod)
    return mod


def test_missing_pyfluidsynth_raises_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "fluidsynth", None)  # makes the import fail
    assert not fluidsynth_available()
    with pytest.raises(ImportError, match="fluid"):
        FluidSynthBackend("guitar.sf2")


def test_config_without_soundfont_is_rejected(cfg, fake_module):
    cfg = cfg.with_overrides({"audio": {"backend": "fluid"}})
    with pytest.raises(ValueError, match="soundfont_path"):
        make_synth(cfg)


def test_unloadable_soundfont_is_rejected(fake_module):
    with pytest.raises(FileNotFoundError):
        FluidSynthBackend("missing.sf2")


def test_strings_map_to_channels(cfg, fake_module):
    cfg = cfg.with_overrides({"audio": {"backend": "fluid", "soundfont_path": "guitar.sf2"}})
    synth = make_synth(cfg)
    assert isinstance(synth, FluidSynthBackend) and isinstance(synth, Synth)
    assert synth.sample_rate == 48_000 and synth.max_block == 128
    fs = synth._fs
    assert fs.samplerate == 48_000.0
    assert fs.calls[0] == ("sfload", "guitar.sf2")
    assert fs.calls[1:7] == [("program", ch, 1, 0, AudioConfig.fluid_program) for ch in range(6)]
    fs.calls.clear()
    for e in (on(0, 40, 90), on(0, 43, 100), on(5, 64, 70), damp(0), damp(0), all_off()):
        synth.handle(e)
    assert fs.calls == [
        ("on", 0, 40, 90),
        ("off", 0, 40),  # restart releases the string's previous note first
        ("on", 0, 43, 100),
        ("on", 5, 64, 70),
        ("off", 0, 43),  # second damp on a released string does nothing
        *[("cc", ch, fluid.CC_ALL_NOTES_OFF, 0) for ch in range(6)],
    ]


def test_render_downmixes_to_mono(fake_module):
    synth = FluidSynthBackend("guitar.sf2")
    out = np.full(128, np.nan)
    synth.render(out)
    np.testing.assert_allclose(out, 0.125)  # (0.5 + -0.25) / 2
    synth.close()
    assert synth._fs.calls[-1] == ("delete",)


def test_real_fluidsynth_plays(cfg):
    pytest.importorskip("fluidsynth")
    path = cfg.audio.soundfont_path
    if not path:
        pytest.skip("no soundfont configured (audio.soundfont_path)")
    synth = FluidSynthBackend(path)
    synth.handle(on(0, 40, 100))
    out = np.zeros(128)
    level = 0.0
    for _ in range(100):
        synth.render(out)
        level = max(level, float(np.abs(out).max()))
    assert np.all(np.isfinite(out)) and level > 0.0
    synth.close()
