"""Settings and the per-user calibration file, both TOML.

``Config`` holds every tunable number in the PID, with the PID's values as defaults.
A TOML file only needs the keys it overrides, e.g.::

    [gate]
    mic = true

    [strings]
    v_min = -0.25

Unknown sections or keys raise ``ValueError`` so typos do not silently fall back to
defaults. ``Calibration`` is what the calibration screen measures for one user; it lives
in its own file so re-running calibration never rewrites hand-edited settings.

Units: times ending in ``_ms`` are milliseconds, racket-frame distances are head
lengths, speeds are head lengths per second, audio sample rates are Hz.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import tomli_w

__all__ = [
    "CameraConfig",
    "TrackerConfig",
    "HandConfig",
    "FilterConfig",
    "StringsConfig",
    "StrumConfig",
    "VelocityConfig",
    "ChordConfig",
    "MuteConfig",
    "GateConfig",
    "MicConfig",
    "AudioConfig",
    "UIConfig",
    "ReplayConfig",
    "ClipConfig",
    "SongConfig",
    "Config",
    "Calibration",
    "home_dir",
    "load_config",
    "save_config",
    "load_calibration",
    "save_calibration",
    "STANDARD_TUNING",
    "DEFAULT_CHORDS",
]

# Low E to high E: E2 A2 D3 G3 B3 E4.
STANDARD_TUNING: tuple[int, ...] = (40, 45, 50, 55, 59, 64)

# Chord voicings, one MIDI note per string (low E first), None = string not played.
# Order is head -> handle end along the handle: section 0 is nearest the head.
DEFAULT_CHORDS: tuple[tuple[str, tuple[int | None, ...]], ...] = (
    ("C", (None, 48, 52, 55, 60, 64)),  # x32010
    ("G", (43, 47, 50, 55, 59, 67)),  # 320003
    ("Am", (None, 45, 52, 57, 60, 64)),  # x02210
    ("F", (41, 48, 53, 57, 60, 65)),  # 133211
)


@dataclass(frozen=True)
class CameraConfig:
    device: int = 0
    fps: int = 60
    fallback_fps: int = 30
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class TrackerConfig:
    """Markerless racket tracking (``vision/flow.py``)."""

    hold_ms: float = 150.0  # keep the last pose this long when tracking is lost


@dataclass(frozen=True)
class HandConfig:
    model_path: str = "models/hand_landmarker.task"
    detect_width: int = 640
    detect_height: int = 360
    num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.3  # lower while tracking, per the risk table
    skip_first_frames_back: int = 1  # ignore a re-acquired hand for strums this many frames
    assoc_max_palms: float = 1.5  # same hand across frames if mean landmark gap is below this


@dataclass(frozen=True)
class FilterConfig:
    """One Euro filter parameters (Casiez et al. 2012), per filtered stream."""

    fingertip_min_cutoff: float = 1.5  # Hz
    fingertip_beta: float = 0.05
    d_cutoff: float = 1.0  # Hz


@dataclass(frozen=True)
class StringsConfig:
    count: int = 6
    u_min: float = 0.15  # along the head, throat (0) -> tip (1)
    u_max: float = 0.85
    v_min: float = -0.30  # across the head, in head lengths
    v_max: float = 0.30
    tuning: tuple[int, ...] = STANDARD_TUNING
    # Which side of v holds string 0 (low E). "auto" puts it on the side that is higher
    # in the image, which matches a guitar held normally.
    low_e_side: Literal["auto", "neg_v", "pos_v"] = "auto"
    auto_side_margin: float = 0.25  # "auto" flips only once |v.y| passes this (~14 degrees)
    head_half_width: float = 0.40  # head ellipse: centre u = 0.5, semi-axes 0.5 and this


@dataclass(frozen=True)
class StrumConfig:
    fingertips: tuple[int, ...] = (8, 12)  # index and middle fingertip landmarks
    min_speed: float = 0.8  # perpendicular, head lengths/s, below this a crossing is ignored
    refractory_ms: float = 40.0  # per string
    finger_dedupe_ms: float = 150.0  # a second fingertip's same-direction pass is a duplicate
    speed_window_ms: float = 70.0  # speed is measured over the current one-direction run
    max_tip_speed: float = 40.0  # faster than this between frames = tracking jump, not strum
    # Only a hand whose palm is within this of the head centre (head lengths), and not
    # on the neck, can be the strumming hand: a fist on the shaft must never arm.
    hand_max_dist: float = 1.0


@dataclass(frozen=True)
class VelocityConfig:
    min_velocity: int = 20
    max_velocity: int = 127
    gamma: float = 1.6
    speed_at_min: float = 0.8  # head lengths/s mapped to min_velocity
    speed_at_max: float = 8.0  # head lengths/s mapped to max_velocity


@dataclass(frozen=True)
class ChordConfig:
    # Handle span along u (negative = below the throat), mapped to t = 0 (head) .. 1 (handle end).
    handle_u_near: float = -0.15
    handle_u_far: float = -1.0
    hysteresis: float = 0.05  # in t units
    knuckle_landmark: int = 9  # middle-finger MCP
    fret_max_v: float = 0.30  # fretting knuckle must be within this of the handle axis
    fret_t_margin: float = 0.05  # ... and at most this far past either end (t units)
    fret_loss_hold_ms: float = 150.0  # keep the chord this long after the hand disappears
    chords: tuple[tuple[str, tuple[int | None, ...]], ...] = DEFAULT_CHORDS


@dataclass(frozen=True)
class MuteConfig:
    max_speed: float = 0.5  # palm speed below this counts as resting, head lengths/s
    hold_ms: float = 150.0  # open palm must rest this long
    damp_ms: float = 30.0  # strings fall silent within this
    open_extension_ratio: float = 1.4  # finger extended if tip/wrist >= this x MCP/wrist


@dataclass(frozen=True)
class GateConfig:
    pick_grip: bool = True  # default gate: closed pick grip = armed, open hand = disarmed
    mic: bool = False  # optional second gate; both on = a crossing needs both
    # Frames a new grip state must persist before it flips; release: before a closed grip
    # disarms. One each, the strict setting chosen with tools/grip_sweep: with the narrow
    # thresholds below, a longer hold only let relaxed returns play (a stray frame can
    # now arm the gate; raise grip_hold_frames to 2 if that shows up).
    grip_hold_frames: int = 1
    grip_release_frames: int = 1
    onset_lookback_ms: float = 80.0  # a crossing plays if an onset came this recently
    onset_after_ms: float = 10.0  # ... or at most this long after it (timing error)
    onset_hold_ms: float = 150.0  # an onset keeps the gate open this long during a sweep
    bleed_ignore_ms: float = 25.0  # ignore onsets this soon after each emitted note
    # Note detection -> sound leaving the speaker; anchors the bleed window until
    # Calibration.detect_to_audio_ms is measured.
    output_latency_ms: float = 30.0
    # Thumb gaps (palm sizes) used until the user calibrates: the template pick grip and
    # spread hand.
    grip_closed_default: float = 0.06
    grip_open_default: float = 0.70
    # Arm below 28 %, disarm above 33 % of the way from the calibrated pick grip to the
    # spread hand; in between the state holds. With the defaults that is a thumb gap of
    # 0.24 / 0.27 palm sizes: the strict setting from tools/grip_sweep on a pick-grip take
    # and a relaxed-return take (2 of 89 relaxed returns played, 25 of 287 strum
    # crossings missed). A thumb even partly off the index reads as not holding a pick.
    grip_arm_below: float = 0.28
    grip_release_above: float = 0.33


@dataclass(frozen=True)
class MicConfig:
    device: int | None = None  # None = system default input
    sample_rate: int = 48_000
    block_size: int = 64  # 1.3 ms at 48 kHz
    band_low_hz: float = 2_000.0
    band_high_hz: float = 8_000.0
    threshold_ratio: float = 3.0  # flux must exceed the noise floor by this factor
    floor_time_constant_ms: float = 500.0  # adaptive noise floor smoothing
    min_interval_ms: float = 20.0  # merge onsets closer than this
    rise_ratio: float = 3.0  # window energy must also exceed the recent peak by this factor
    warmup_ms: float = 100.0  # no onsets until the floor has seen this much audio
    max_active_ms: float = 300.0  # above threshold longer than this = new noise level


@dataclass(frozen=True)
class AudioConfig:
    backend: Literal["karplus", "fluid"] = "karplus"
    device: int | None = None  # None = system default output
    sample_rate: int = 48_000
    block_size: int = 128  # 2.7 ms at 48 kHz
    soundfont_path: str | None = None
    master_gain: float = 0.8
    ks_decay: float = 0.996  # Karplus-Strong loop gain per period
    ks_brightness_min: float = 0.2  # lowpass mix at velocity 1
    ks_brightness_max: float = 0.9  # lowpass mix at velocity 127
    voice_gain: float = 0.35  # per-voice scale, so a full 6-string chord stays near unity
    dark_cutoff_hz: float = 600.0  # one-pole lowpass of the dark excitation component
    damp_fade_fraction: float = 2 / 3  # damp fade spans this much of MuteConfig.damp_ms
    clip_knee: float = 0.8  # soft clipper is the identity below this
    fluid_program: int = 25  # General MIDI "Acoustic Guitar (steel)", 0-based
    fluid_gain: float = 0.4  # FluidSynth's own output gain


@dataclass(frozen=True)
class UIConfig:
    mirror: bool = True
    max_fps: int = 30
    flash_ms: float = 300.0
    trail_frames: int = 8


@dataclass(frozen=True)
class ReplayConfig:
    record_dir: str = "recordings"
    # Also record lossless camera video, for --from-video replays. Off by default: it is
    # about 0.8 GB a minute and can drop frames; the recorded data replays exactly anyway.
    video: bool = False


@dataclass(frozen=True)
class ClipConfig:
    """F8 clip snapping (``replay/clip.py``): the last ``seconds`` of the window and sound."""

    seconds: float = 30.0  # how much history F8 saves
    folder: str = "memories"  # relative paths are under the repo root
    # The window frames are kept in memory as JPEGs: about 65 KB each at 720p, so the default
    # 30 s at 30 fps is roughly 60 MB. Lower this to trade picture quality for memory.
    jpeg_quality: int = 80


@dataclass(frozen=True)
class SongConfig:
    """Song mode (``song/``): a chord sheet drives which chords sit on the handle."""

    speed: float = 0.7  # play songs at this fraction of the sheet's tempo
    count_in_beats: int = 4  # "3, 2, 1, GO" before the first chord
    chord_bars: float = 1.0  # length of a chord the sheet gives no length for
    simplify_slash: bool = True  # play C/B as C, so passing bass notes use no section
    listen_s: float = 3.0  # push-to-talk recording length
    stt_model: str = "scribe_v2"  # ElevenLabs speech-to-text model
    stt_timeout_s: float = 30.0
    match_min_score: float = 0.5  # a spoken title must match a song file at least this well


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackerConfig = field(default_factory=TrackerConfig)
    hands: HandConfig = field(default_factory=HandConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    strings: StringsConfig = field(default_factory=StringsConfig)
    strum: StrumConfig = field(default_factory=StrumConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    chords: ChordConfig = field(default_factory=ChordConfig)
    mute: MuteConfig = field(default_factory=MuteConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    mic: MicConfig = field(default_factory=MicConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    song: SongConfig = field(default_factory=SongConfig)

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with nested ``{"section": {"key": value}}`` overrides applied."""
        return _merge(self, overrides, "")


@dataclass(frozen=True)
class Calibration:
    """Per-user measurements from the calibration screen. None = not measured yet.

    The grip scores are whatever scalar ``play/gate.py`` computes from a hand (lower =
    more closed); the gate derives its hysteresis thresholds from the two means.
    """

    grip_closed_score: float | None = None
    grip_open_score: float | None = None
    mic_noise_floor: float | None = None
    mic_brush_level: float | None = None
    detect_to_audio_ms: float | None = None  # last measured internal latency
    grip_version: int | None = None  # play.gate.GRIP_SCORE_VERSION the grip scores used


def home_dir() -> Path:
    """Where user files live: ``$ANYSTRING_HOME`` or ``~/.anystring``.

    The project was first called Racket Guitar; an existing ``~/.racketguitar`` is kept in
    use until ``~/.anystring`` exists, so old calibration and songs are not lost.
    """
    env = os.environ.get("ANYSTRING_HOME")
    if env:
        return Path(env)
    new = Path.home() / ".anystring"
    legacy = Path.home() / ".racketguitar"
    return legacy if legacy.is_dir() and not new.exists() else new


def load_config(path: str | Path | None = None) -> Config:
    """Load settings, or defaults when no path is given and ``config.toml`` is absent."""
    p = Path(path) if path is not None else home_dir() / "config.toml"
    if not p.exists():
        if path is not None:
            raise FileNotFoundError(p)
        return Config()
    with p.open("rb") as f:
        return Config().with_overrides(tomllib.load(f))


def save_config(cfg: Config, path: str | Path) -> None:
    _write_toml(_to_toml(asdict(cfg)), Path(path))


def load_calibration(path: str | Path | None = None) -> Calibration:
    p = Path(path) if path is not None else home_dir() / "calibration.toml"
    if not p.exists():
        return Calibration()
    with p.open("rb") as f:
        return _merge(Calibration(), tomllib.load(f), "")


def save_calibration(cal: Calibration, path: str | Path | None = None) -> None:
    p = Path(path) if path is not None else home_dir() / "calibration.toml"
    _write_toml(_to_toml(asdict(cal)), p)


def _merge(obj: Any, overrides: dict[str, Any], where: str) -> Any:
    known = {f.name: f for f in fields(obj)}
    changes: dict[str, Any] = {}
    for key, value in overrides.items():
        name = f"{where}{key}"
        if key not in known:
            raise ValueError(f"unknown config key: {name}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a table")
            changes[key] = _merge(current, value, f"{name}.")
        else:
            changes[key] = _from_toml(value)
    return replace(obj, **changes)


def _from_toml(value: Any) -> Any:
    # TOML has arrays, not tuples, and no null; "none" stands in for None inside arrays.
    if isinstance(value, list):
        return tuple(_from_toml(v) for v in value)
    if value == "none":
        return None
    return value


def _to_toml(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_toml(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [("none" if v is None else _to_toml(v)) for v in value]
    return value


def _write_toml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        tomli_w.dump(data, f)
    tmp.replace(path)
