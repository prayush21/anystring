# Racket Guitar

A racket (tennis or badminton) played as a guitar using only a MacBook camera and mic,
with nothing attached to the racket. The full spec is [docs/PID.md](docs/PID.md); read
it before changing behaviour, but note its printed-marker parts are superseded (below). This file is the
short version plus the rules for working in the repo.

## Commands

```bash
uv sync                      # install (Python 3.12, deps pinned in uv.lock)
uv run pytest                # headless suite; hardware tests are deselected
uv run pytest -m hardware    # camera/mic/speaker tests, MacBook only
uv run ruff check . && uv run ruff format .
uv run racketguitar fetch-model   # once: MediaPipe hand model into models/
uv run racketguitar          # play (calibrates on first run); `calibrate`, `replay DIR`
uv run python -m tools.latency_probe analyze take.wav
```

## Architecture

Four threads pass small timestamped messages through queues, so no slow step can
delay a note:

1. **Camera** (`capture/camera.py`): 60 fps capture (30 fallback), buffer size 1, newest
   frame only, stamped with `time.monotonic()` at capture.
2. **Vision** (`vision/`, `geometry/`, `play/`): MediaPipe Hand Landmarker in VIDEO
   mode on a 640x360 copy -> `HandObs` (One Euro filtered); the markerless racket
   tracker (`vision/flow.py`) -> `RacketPose`; then strum, gate, chord and mute logic
   emit `NoteEvent`s.
3. **Mic** (`capture/mic.py`): 48 kHz, 64-sample blocks; high-band spectral-flux
   onsets -> `Onset`. Optional, off by default.
4. **Audio** (`audio/`): 48 kHz, 128-sample output callback rendering one
   Karplus-Strong voice per string. The callback allocates nothing and takes no locks.

The main thread only draws the overlay (`ui/`), because macOS needs GUI work there.

**Racket tracking is markerless.** There are no printed markers (the ArUco version was
removed; the PID's marker sections are superseded). The player clicks four points in
the window: the top of the head (tip), the throat (head meets shaft), where the grip
starts, and the butt. `FlowTracker` then follows corner features on a ring around the
head's rim with pyramidal Lucas-Kanade (forward-backward checked) and one RANSAC
similarity transform per frame, moving the tip and throat that define the
`RacketPose`. The grip and butt clicks set `chords.handle_u_near/far`, so the four
chord sections cover the grip only. Known limits: fast moves can drift (re-lock with
`l`), and it cannot tell the racket left the frame (the static background keeps
"tracking"). Its one setting is `[tracking] hold_ms`.
No network or AI call ever sits between a strum and its sound.

Play rules (decided, do not reopen without the lead):

- **Strum gate** defaults to the **pick grip**: closed pick grip (thumb pad on the side
  of a curled index, other fingers curled) = armed, open hand = disarmed. Per-user
  calibration, hysteresis, 2-frame hold. The **mic gate** is optional and off by
  default; with both on, a crossing needs a pick grip *and* an onset in the last 80 ms.
- **Chords** come only from which of **4 handle sections** the fretting hand grips:
  C, G, Am, F from head to butt. There is no finger chord-shape recognition. The
  fretting hand is whichever hand is nearer the handle; no hand = open strings.
- **Strings**: 6 segments parallel to u, at u 0.15..0.85, spread across v -0.30..+0.30
  head lengths. Each string has a 40 ms refractory window, with no global cooldown.
- **Palm mute**: open strumming hand resting on the head for 150 ms damps all strings.

## Contracts (`racketguitar/types.py`, frozen)

| Type | Producer -> consumer | Meaning |
| --- | --- | --- |
| `Vec2(x, y)` | everywhere | pixel point or vector (a NamedTuple) |
| `RacketPose` | WP1 -> WP2 | `origin` = throat point, `u` = unit throat->tip, `v` = perpendicular, `head_len` px (throat to tip) |
| `HandObs` | WP1 -> WP2 | `(21, 2)` filtered `landmarks` and unfiltered `raw`, pixels |
| `Onset` | WP4 -> gate | mic contact onset, `strength` = ratio over noise floor |
| `StringHit` | inside WP2 | string 0 (low E)..5, `"down"`/`"up"`, speed in head lengths/s |
| `NoteEvent` | WP2 -> WP3 | `on` / `damp` / `all_off`, MIDI note, velocity 1..127 |
| `StrumGate` | WP4 -> WP2 | Protocol: `observe_hand`, `observe_onset`, `note_emitted`, `allows`, `grip` |

Conventions all packages share (the docstring in `types.py` is the source of truth):

- `t` is `time.monotonic()` seconds; for camera data it is the **capture** time.
- Pixels are in the **full-resolution, un-mirrored** camera frame. Detectors that run
  on a downscaled copy scale back. Only the overlay mirrors.
- `v = (-u.y, u.x)`. Racket coords are in head lengths:
  `px = origin + (u*U + v*V) * head_len`.
- "down" = moving from the low-E side toward high E.
- `play/engine.py` receives its `StrumGate` by injection and never imports
  `play/gate.py`.

All thresholds live in `racketguitar/config.py` (`Config`, TOML overrides, unknown keys
rejected); per-user measurements live in `Calibration` (`~/.racketguitar/`, or
`$RACKETGUITAR_HOME`). Do not hard-code a tuning number: add a `Config` key (via the
lead) and read it. `tests/test_config_wiring.py` fails if a key is never read.

## The import rule

**No package imports another package except `types.py` and `config.py`.** A work
package may import `racketguitar.types`, `racketguitar.config`, its own modules,
third-party libraries and the stdlib, and nothing else. `tests/test_import_boundaries.py`
enforces this with an ownership map:

| WP | Owns |
| --- | --- |
| WP0 (lead) | `types.py`, `config.py`, package `__init__`s, `tests/fakes/`, top-level tests |
| WP1 Vision | `capture/camera.py`, `vision/` (hands, filters, markerless `flow.py`), `tests/vision/` |
| WP2 Play logic | `geometry/`, `play/strum.py`, `play/chords.py`, `play/mute.py`, `play/engine.py`, `tests/play/` |
| WP3 Audio | `audio/`, `tests/audio/` |
| WP4 Contact gate | `capture/mic.py`, `play/gate.py`, `tests/gate/` |
| WP5 Integration | `app.py`, `pipeline.py`, `ui/`, `replay/`, `tools/latency_probe.py`, `tests/integration/` (the only layer allowed to import across packages) |
| WP7 Song mode | `song/` (sheets, voicings, player, library, voice), `tests/song/` |

`play/` and `capture/` are shared directories: WP2 and WP4 both live in `play/`, WP1
and WP4 in `capture/`, and they still must not import each other. A new file there
needs an owner added to the map by the lead. Changes to `types.py`, `config.py`,
`pyproject.toml` or the fakes also go through the lead; request them in your report
rather than editing.

## Testing

- Every test runs **headless**: no camera, mic, speakers or network. Anything that
  needs hardware is `@pytest.mark.hardware`, and must never be the only proof that
  something works.
- Use the fakes in `tests/fakes/` as ground truth: `make_pose`, `uv_to_px`/`px_to_uv`,
  `hand_template`/`place_hand` (`open`, `pick`, `fist`), `strum_track`,
  `fretting_hand`, `expected_slots_crossed`, `brush_strokes`, `room_noise`,
  `speaker_bleed`, `blocks`, `render_racket` (a plain textured racket on a static
  background, with exact tip and throat), `ScriptedGate`, `FakeClock`. Production code
  never imports them.
- Fixtures in `tests/conftest.py`: `rng` (seeded), `cfg` (defaults with
  `low_e_side="neg_v"`), and an autouse temp `RACKETGUITAR_HOME`.
- Tests must be deterministic: seed every RNG, and never sleep or read the wall clock.

## Gotchas

- The package has a module named `types.py`. Always import it as
  `racketguitar.types`, and never run a file inside `racketguitar/` as a script (use
  `python -m`), or it shadows the stdlib `types`.
- `HandObs` holds numpy arrays, so `==` on whole instances is not meaningful. Compare
  fields with `np.array_equal`.
- mediapipe brings `opencv-contrib-python`. Never add `opencv-python` too, because
  both install the same `cv2` module.
- The MediaPipe hand model (`models/hand_landmarker.task`) is downloaded, not committed.

## Phases

Work is gated: no phase starts until the lead approves the previous gate. Phase 0 is
WP0 (this skeleton). Phase 1 is WP1 to WP4 in parallel, each built against only its
own row of the PID's work-package table. Next comes WP5 integration, then on-device
acceptance on the MacBook, then the WP6 verification by a fresh agent.

## Song mode (WP7)

- A chord sheet (ChordPro or chords above lyrics, in `~/.racketguitar/songs/`, never
  in the repo since sheets hold lyrics) becomes a timeline. `SongPlayer` plans which
  chord labels each handle section over time; chords still come only from grip
  position. Relabels reach the engine only via `FrameInput.chords` ->
  `Engine.set_chords`, which keeps gate, strum and mute state, so recordings replay
  exactly and a relabel on the beat drops nothing.
- Voice (`s`): a short clip goes to ElevenLabs speech-to-text (`$ELEVENLABS_API_KEY`)
  on a worker thread; titles match song files fuzzily. No network call is on the play
  path. Tests use made-up lyrics only.

## Integration (WP5)

- `pipeline.Pipeline.step(FrameInput)` is the one per-frame play step: drain onsets into
  the gate, then run the engine. The live app and the replay harness both call it, so
  a replay proves the live behaviour. Never add play logic outside it.
- Replay feeds a session's recorded observations (poses, hands, drained onsets) back
  through the pipeline and must match exactly; `--from-mic` re-detects onsets from
  `mic.wav`. Poses always replay as recorded (the tracker needs the lock clicks).
  `[replay] video = true` also saves lossless FFV1 video, only for inspecting a take.
  Changing settings mid-recording (calibration, re-lock, gate toggles) updates the
  recording or splits it into `-partN`, so every recording replays exactly.
  `tests/integration/song.py` is a scripted four-chord performance used end to end.
- `audio/probe.py` (`--probe-tone`) plays a 1 kHz beep per note, so
  `tools/latency_probe.py` can measure tap-to-sound latency from a phone recording.
