"""Optional FluidSynth (SoundFont) backend behind the ``Synth`` interface.

Needs the ``fluid`` extra (pyfluidsynth, imported lazily as ``fluidsynth``), the
FluidSynth C library, and ``AudioConfig.soundfont_path``. String ``s`` plays on MIDI
channel ``s``, so each string keeps one note like the Karplus-Strong voices: ``on``
releases the string's previous note before starting the new one, ``damp`` releases it,
and ``all_off`` sends all-notes-off on every string channel.

Unlike the Karplus-Strong synth this backend is not allocation-free: pyfluidsynth's
``get_samples`` returns a new array per call. It is an optional sound, not the
real-time default.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from racketguitar.config import AudioConfig, Config
from racketguitar.types import NUM_STRINGS, NoteEvent

__all__ = ["FluidSynthBackend", "fluidsynth_available"]

FLUID_BANK = 0
CC_ALL_NOTES_OFF = 123


def _import_fluidsynth() -> Any:
    try:
        return importlib.import_module("fluidsynth")
    except (ImportError, OSError) as e:  # OSError: module present, C library missing
        raise ImportError(
            "the FluidSynth backend needs pyfluidsynth and the FluidSynth library "
            "(install the 'fluid' extra and `brew install fluid-synth`)"
        ) from e


def fluidsynth_available() -> bool:
    try:
        _import_fluidsynth()
    except ImportError:
        return False
    return True


class FluidSynthBackend:
    """SoundFont playback with one MIDI channel per string."""

    def __init__(
        self,
        soundfont_path: str,
        sample_rate: int = 48_000,
        max_block: int = 128,
        *,
        program: int = AudioConfig.fluid_program,
        bank: int = FLUID_BANK,
        gain: float = AudioConfig.fluid_gain,
    ) -> None:
        fluidsynth = _import_fluidsynth()
        self._sr = int(sample_rate)
        self._max_block = int(max_block)
        self._fs = fluidsynth.Synth(gain=gain, samplerate=float(sample_rate))
        sfid = self._fs.sfload(str(soundfont_path))
        if sfid == -1:
            raise FileNotFoundError(f"could not load soundfont: {soundfont_path}")
        for ch in range(NUM_STRINGS):
            self._fs.program_select(ch, sfid, bank, program)
        self._held: list[int | None] = [None] * NUM_STRINGS

    @classmethod
    def from_config(cls, cfg: Config) -> FluidSynthBackend:
        a = cfg.audio
        if not a.soundfont_path:
            raise ValueError("audio.backend = 'fluid' needs audio.soundfont_path")
        return cls(
            a.soundfont_path,
            a.sample_rate,
            a.block_size,
            program=a.fluid_program,
            gain=a.fluid_gain,
        )

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def max_block(self) -> int:
        return self._max_block

    def handle(self, event: NoteEvent) -> None:
        if event.kind == "on":
            self._release(event.string)
            self._fs.noteon(event.string, event.midi, event.velocity)
            self._held[event.string] = event.midi
        elif event.kind == "damp":
            self._release(event.string)
        elif event.kind == "all_off":
            for ch in range(NUM_STRINGS):
                self._held[ch] = None
                self._fs.cc(ch, CC_ALL_NOTES_OFF, 0)

    def _release(self, ch: int) -> None:
        held = self._held[ch]
        if held is not None:
            self._fs.noteoff(ch, held)
            self._held[ch] = None

    def render(self, out: np.ndarray) -> None:
        n = out.shape[0]
        s = np.asarray(self._fs.get_samples(n), dtype=np.float64).reshape(n, 2)
        np.add(s[:, 0], s[:, 1], out=out)
        out *= 0.5 / 32768.0

    def reset(self) -> None:
        for ch in range(NUM_STRINGS):
            self._held[ch] = None
            self._fs.cc(ch, CC_ALL_NOTES_OFF, 0)

    def close(self) -> None:
        self._fs.delete()
